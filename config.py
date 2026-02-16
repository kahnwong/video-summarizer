from types import SimpleNamespace

audio_chunks_dir = "data/temp/audio_chunks"
transcript_file = "data/output/transcript.json"

split_audio = SimpleNamespace(
    INPUT_FILE="data/source/audio.wav",
    OUTPUT_DIR=audio_chunks_dir,
    MIN_SILENCE_LEN=1500,
    SILENCE_THRESH=-60,
    MAX_CHUNK_MS=10 * 60 * 1000,
)

asr = SimpleNamespace(
    DIAR_MODEL_NAME="pyannote/speaker-diarization-3.0",
    ASR_MODEL_NAME="scb10x/typhoon-asr-realtime",
    AUDIO_DIR=audio_chunks_dir,
    TEMP_FILE="data/temp/temp.wav",
    OUTPUT_DIR="data/output",
    OUTPUT_FILE=transcript_file,
)

summarize = SimpleNamespace(
    TRANSCRIPT_FILE=transcript_file,
    TEMPLATE_FILE="templates/summarize_prompt.jinja2",
    MODEL_NAME="claude-sonnet-4-5",
    OUTPUT_FILE="data/output/summary.json",
    MAX_IMAGES=10,
)
