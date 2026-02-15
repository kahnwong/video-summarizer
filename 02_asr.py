import json
import logging
import os
from pathlib import Path

import librosa
import soundfile as sf
import torch
from loguru import logger as log
from pyannote.audio import Pipeline

# ----- Suppress NeMo warnings -----
os.environ["NEMO_LOG_LEVEL"] = "ERROR"

# Import NeMo logging first and apply filter before importing ASR models
from nemo.utils import logging as nemo_logging


class NeMoWarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Suppress specific NeMo warnings
        if "Lhotse dataloader" in msg:
            return False
        if "non-tarred dataset" in msg:
            return False
        if "Megatron" in msg:
            return False
        return True


# Set log level and add filter BEFORE importing nemo ASR
nemo_logging.setLevel(nemo_logging.ERROR)
if hasattr(nemo_logging, "_logger") and nemo_logging._logger is not None:
    for handler in nemo_logging._logger.handlers:
        handler.addFilter(NeMoWarningFilter())

# Now import nemo ASR with filters in place
import nemo.collections.asr as nemo_asr

# ----- Python 3.13 compatibility patch for lhotse -----
# Fix for: TypeError: object.__init__() takes exactly one argument
from torch.utils.data import Sampler

from config import asr

_original_sampler_init = Sampler.__init__


def _patched_sampler_init(self, data_source=None):
    """Patched Sampler.__init__ that doesn't pass arguments to object.__init__()"""
    # In Python 3.13, object.__init__() doesn't accept arguments
    # Just skip calling super().__init__() since Sampler directly inherits from object
    pass


Sampler.__init__ = _patched_sampler_init

# ----- init -----
os.makedirs(asr.OUTPUT_DIR, exist_ok=True)


# ----- functions -----
def load_models():
    """Load speaker diarization and ASR models"""
    log.info("Loading speaker diarization model")

    # Get HuggingFace token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log.warning(
            "HF_TOKEN not found in environment. If model is gated, this will fail."
        )
        log.warning("To fix: export HF_TOKEN='your_token_here'")
        log.warning("Get token from: https://huggingface.co/settings/tokens")
        log.warning(
            "Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.0"
        )

    diarization_pipeline = Pipeline.from_pretrained(asr.DIAR_MODEL_NAME, token=hf_token)

    log.info("Loading ASR model")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(asr.ASR_MODEL_NAME)

    return diarization_pipeline, asr_model


def process_audio_file(audio_path: str, diarization_pipeline, asr_model, output_file):
    """Process a single audio file with diarization and ASR"""
    log.info(f"Processing {audio_path}")

    # Load audio using librosa as mono (since torchcodec is not available)
    log.info("Loading audio file")
    waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    # Convert to torch tensor and add channel dimension for pyannote
    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)  # Shape: (1, samples)

    # Create audio dict for pyannote
    audio_dict = {"waveform": waveform_tensor, "sample_rate": sample_rate}

    # Run speaker diarization
    log.info("Running speaker diarization")
    diarization_output = diarization_pipeline(audio_dict)

    # Extract results by transcribing each speaker segment separately
    audio_filename = os.path.basename(audio_path)
    segment_count = 0

    # DiarizeOutput has speaker_diarization attribute which contains the annotation
    if hasattr(diarization_output, "speaker_diarization"):
        annotation = diarization_output.speaker_diarization

        for turn, _, speaker_label in annotation.itertracks(yield_label=True):
            # Extract audio segment for this speaker turn
            start_sample = int(turn.start * sample_rate)
            end_sample = int(turn.end * sample_rate)
            segment_waveform = waveform[start_sample:end_sample]

            # Save segment to temporary file
            temp_segment_path = asr.TEMP_FILE
            sf.write(temp_segment_path, segment_waveform, sample_rate)

            # Transcribe this specific segment
            # log.info(
            #     f"Transcribing segment {turn.start:.2f}s - {turn.end:.2f}s for {speaker_label}"
            # )
            transcription_result = asr_model.transcribe(
                [temp_segment_path], batch_size=1
            )

            # Extract text from result
            if transcription_result:
                result = transcription_result[0]
                if hasattr(result, "text"):
                    segment_text = result.text
                else:
                    segment_text = str(result)
            else:
                segment_text = ""

            # Create result object
            result_obj = {
                "speaker": speaker_label,
                "word": segment_text,
                "start": turn.start * 1000,
                "end": turn.end * 1000,
                "audio": audio_filename,
            }

            # Write to file as NDJSON (one JSON object per line)
            output_file.write(json.dumps(result_obj, ensure_ascii=False) + "\n")
            output_file.flush()  # Ensure it's written to disk
            segment_count += 1
    else:
        log.error("Unable to extract segments from diarization output")

    return segment_count


def main():
    log.info("Starting ASR processing")

    # Load models
    diarization_pipeline, asr_model = load_models()

    # Get all audio files
    audio_files = sorted(Path(asr.AUDIO_DIR).glob("*.mp3"))
    log.info(f"Found {len(audio_files)} audio files")

    # Open output file and write results incrementally
    log.info(f"Writing results to {asr.OUTPUT_FILE}")
    total_segments = 0

    with open(asr.OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Process each file
        for audio_file in audio_files:
            segment_count = process_audio_file(
                str(audio_file), diarization_pipeline, asr_model, f
            )
            total_segments += segment_count

    log.info(f"Done! Processed {total_segments} segments")


if __name__ == "__main__":
    main()
