import json
from typing import Any, Dict

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from loguru import logger as log
from numpy import dtype, float64, ndarray
from pyannote.audio import Pipeline
from pydub import AudioSegment
from speechbrain.pretrained import EncoderClassifier
import os
from config import asr
import soundfile as sf

# ----- init -----
os.makedirs(asr.OUTPUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- init models -----
log.info("Loading diarization model...")

diar_pipeline = Pipeline.from_pretrained(asr.DIAR_MODEL_NAME)
log.info("Loading ASR model...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name=asr.ASR_MODEL_NAME, map_location=device
)

log.info("Loading speaker embedding model...")
spk_model = EncoderClassifier.from_hparams(
    source=asr.SPEAKER_EMBEDDING_MODEL_NAME, run_opts={"device": device}
)


# ----- utils -----
def load_audio(path: str) -> tuple[ndarray[tuple[Any, ...], dtype[float64]], int]:
    """Loads audio and convert to wav 16k mono internally."""
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples, 16000


def get_embedding(waveform: ndarray[tuple[Any, ...], dtype[float64]]) -> Any:
    """Extract speaker embedding vector."""
    log.info(waveform.shape)
    if waveform.ndim == 1:
        waveform = torch.tensor(waveform).unsqueeze(0)

    return spk_model.encode_batch(waveform.to(device)).detach().cpu().numpy()[0]


def match_speaker(global_db: Dict[str, Any], emb_vec: Any):
    def cosine(a, b):
        return np.dot(a.flatten(), b.flatten()) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

    """Match embedding to global speakers DB."""
    if len(global_db) == 0:
        return None

    best_id, best_score = None, -1
    for spk_id, vecs in global_db.items():
        avg_vec = np.mean(vecs, axis=0)
        score = cosine(avg_vec, emb_vec)
        if score > best_score:
            best_id = spk_id
            best_score = score

    return best_id if best_score >= asr.SIM_THRESHOLD else None


def merge_output(results):
    with open(asr.OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"Saved: {asr.OUTPUT_FILE}")


# ----- pipeline -----
def pipeline():
    audio_files = sorted(
        [
            os.path.join(asr.AUDIO_DIR, f)
            for f in os.listdir(asr.AUDIO_DIR)
            if f.lower().endswith((".wav", ".mp3"))
        ]
    )

    GLOBAL_SPK_DB = {}  # { "S1": [emb, emb, ...], ... }
    global_speaker_count = 1
    abs_time = 0
    output = []

    for file in audio_files:
        log.info(f"Processing: {file}")

        wav, sample_rate = load_audio(file)

        diar_output = diar_pipeline(
            {"waveform": torch.tensor(wav).unsqueeze(0), "sample_rate": sample_rate}
        )

        for turn, speaker in diar_output.speaker_diarization:
            try:
                # ----- match speaker id -----
                seg_wav = wav[int(turn.start * sample_rate) : int(turn.end * sample_rate)]
                emb = get_embedding(seg_wav)

                match_id = match_speaker(GLOBAL_SPK_DB, emb)
                log.info(f"  Local speaker: {speaker} --> Global speaker: {match_id}")

                if match_id is None:
                    match_id = f"S{global_speaker_count}"
                    GLOBAL_SPK_DB.setdefault(match_id, []).append(emb)
                    global_speaker_count += 1
                    log.info(f"    New global speaker created: {match_id}")
                else:
                    GLOBAL_SPK_DB[match_id].append(emb)
                    log.info(f"    Matched with existing global speaker: {match_id}")

                sf.write(asr.TEMP_FILE, seg_wav, sample_rate)

                # ----- append output -----
                try:
                    # Pass audio data directly instead of file path to avoid dataloader issues
                    transcribe_result = asr_model.transcribe(
                        audio=asr.TEMP_FILE, batch_size=1, timestamps=True
                    )
                    # transcribe_result[0] is list of hypotheses for first file
                    # transcribe_result[0][0] is the best hypothesis
                    hypothesis = transcribe_result[0][0]
                    texts = hypothesis.timestep["word"]
                    for w in texts:
                        output.append(
                            {
                                "speaker": match_id,
                                "word": w["word"],
                                "start": float(turn.start) + abs_time + w["start"],
                                "end": float(turn.end) + abs_time + w["end"],
                                "audio": file,
                            }
                        )
                except Exception as e2:
                    log.error(f"    Transcription error: {type(e2).__name__}: {e2}")
                    # Skip this segment and continue
                    continue
                # transcribe_result[0] is list of hypotheses for first file
                # transcribe_result[0][0] is the best hypothesis
                hypothesis = transcribe_result[0][0]
                texts = hypothesis.timestep["word"]
                for w in texts:
                    output.append(
                        {
                            "speaker": match_id,
                            "word": w["word"],
                            "start": float(turn.start) + abs_time + w["start"],
                            "end": float(turn.end) + abs_time + w["end"],
                            "audio": file,
                        }
                    )
            except Exception as e:
                log.error(f"    Error processing segment: {e}")
                continue

        abs_time += len(wav) / sample_rate

    return output


if __name__ == "__main__":
    result = pipeline()
    merge_output(result)
