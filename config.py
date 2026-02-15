from types import SimpleNamespace

audio_chunks_dir = "data/temp/audio_chunks"

split_audio = SimpleNamespace(
    INPUT_FILE="data/source/audio.wav",
    OUTPUT_DIR=audio_chunks_dir,
    MIN_SILENCE_LEN=1500,
    SILENCE_THRESH=-60,
    MAX_CHUNK_MS=10 * 60 * 1000,
)

asr = SimpleNamespace(
    ASR_MODEL_NAME="scb10x/typhoon-asr-realtime",
    DIAR_MODEL_NAME="pyannote/speaker-diarization-community-1",
    SPEAKER_EMBEDDING_MODEL_NAME="speechbrain/spkrec-ecapa-voxceleb",
    AUDIO_DIR=audio_chunks_dir,
    SIM_THRESHOLD=0.75,  # higher = stricter matching
    OUTPUT_DIR="data/output",
    OUTPUT_FILE="data/output/transcript.json",
)
