from types import SimpleNamespace

split_audio = SimpleNamespace(
    INPUT_FILE="data/source/audio.wav",
    OUTPUT_DIR="data/temp/audio_chunks",
    MIN_SILENCE_LEN=1500,
    SILENCE_THRESH=-60,
    MAX_CHUNK_MS=10 * 60 * 1000,
)
