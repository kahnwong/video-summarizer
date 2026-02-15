from typing import Any

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
#
#
# def cosine(a, b):
#     return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b))
#
#
# def match_speaker(global_db, emb_vec):
#     """Match embedding to global speakers DB."""
#     if len(global_db) == 0:
#         return None
#
#     best_id, best_score = None, -1
#     for spk_id, vecs in global_db.items():
#         avg_vec = np.mean(vecs, axis=0)
#         score = cosine(avg_vec, emb_vec)
#         if score > best_score:
#             best_id = spk_id
#             best_score = score
#
#     return best_id if best_score >= SIM_THRESHOLD else None
#
#
# # ============================
# # MAIN PIPELINE
# # ============================

# ----- pipeline functions -----


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
    full_output = []

    for file in audio_files:
        log.info(f"Processing: {file}")

        wav, sr = load_audio(file)

        # ---- 1. DIARIZATION ----
        diar_output = diar_pipeline(
            {"waveform": torch.tensor(wav).unsqueeze(0), "sample_rate": sr}
        )

        for turn, speaker in diar_output.speaker_diarization:
            try:
                seg_wav = wav[int(turn.start * sr): int(turn.end * sr)]
                emb = get_embedding(seg_wav)

                # # ---- 2. GLOBAL SPEAKER MATCHING ----
                # match_id = match_speaker(GLOBAL_SPK_DB, emb)
                # print(f"  Local speaker: {spk} --> Global speaker: {match_id}")
                #
                # if match_id is None:
                #     match_id = f"S{global_speaker_count}"
                #     GLOBAL_SPK_DB.setdefault(match_id, []).append(emb)
                #     global_speaker_count += 1
                #     print(f"    New global speaker created: {match_id}")
                # else:
                #     GLOBAL_SPK_DB[match_id].append(emb)
                #     print(f"    Matched with existing global speaker: {match_id}")
                #
                # sf.write("tmp.wav", seg_wav, sr)
                # texts = asr_model.transcribe("tmp.wav", timestamps=True)[0].timestamp[
                #     "word"
                # ]
                # for w in texts:
                #     # print(f"start: {seg['start']}, {w['start']}, end: {seg['end']}, {w['end']}")
                #     full_output.append(
                #         {
                #             "speaker": match_id,
                #             "word": w["word"],
                #             "start": float(turn.start) + abs_time + w["start"],
                #             "end": float(turn.end) + abs_time + w["end"],
                #             "audio": file,
                #         }
                #     )
            except Exception as e:
                print("    Error processing segment:", e)
                continue

        abs_time += len(wav) / sr
        print("full_output :", full_output)

    return full_output


#
# # ============================
# # MERGE TO FINAL JSON
# # ============================
# def merge_to_json(results):
#     with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
#
#     print("Saved:", OUTPUT_JSON)
#
#
if __name__ == "__main__":
    result = pipeline()
#     merge_to_json(result)
#     print("Done!")
