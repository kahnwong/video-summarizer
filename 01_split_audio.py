import math
import os
from typing import List

from loguru import logger as log
from pydub import AudioSegment, silence
from tqdm import tqdm

# ----- config -----
INPUT_FILE = "data/source/audio.wav"
OUTPUT_DIR = "data/temp/audio_chunks"
MIN_SILENCE_LEN = 1500  # 1.5s
SILENCE_THRESH = -60  # dBFS
MAX_CHUNK_MS = 10 * 60 * 1000  # 5 minutes

# ----- init -----
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----- functions -----
def create_split_points(audio: AudioSegment) -> List[int]:
    log.info("Creating split points")

    # Detect silence ranges
    silences = silence.detect_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        seek_step=10,  # 10ms
    )

    # Convert to simple list of silence split points
    split_points = [s[0] for s in silences]
    split_points.append(len(audio))  # last segment end

    return split_points


def split_by_silence(
    audio: AudioSegment, split_points: List[int]
) -> List[AudioSegment]:
    log.info("Splitting audio by silence")

    segments = []
    start = 0

    print("# Split by silence")
    for end in tqdm(split_points):
        if end - start > 1000:  # ignore segments < 1 sec
            segments.append(audio[start:end])
        start = end

    return segments


def split_chunks(segments: List[AudioSegment]) -> List[AudioSegment]:
    # 5 mins per chunk max
    log.info("Split into 5min chunks")
    final_chunks = []

    for seg in segments:
        if len(seg) <= MAX_CHUNK_MS:
            final_chunks.append(seg)
        else:
            # further subdivide long segments
            parts = math.ceil(len(seg) / MAX_CHUNK_MS)
            for i in range(parts):
                sub = seg[i * MAX_CHUNK_MS : (i + 1) * MAX_CHUNK_MS]
                final_chunks.append(sub)

    return final_chunks


def conver_to_mp3(chunks: List[AudioSegment]) -> None:
    log.info("Converting to mp3")

    for i, chunk in enumerate(tqdm(chunks)):
        out_path = os.path.join(OUTPUT_DIR, f"chunk_{i:03d}.mp3")
        chunk.export(out_path, format="mp3")

    log.info(f"Done! Total chunks: {len(chunks)}")


if __name__ == "__main__":
    log.info("Creating audio segments")
    audio = AudioSegment.from_file(INPUT_FILE)

    split_points = create_split_points(audio)
    segments = split_by_silence(audio, split_points)
    chunks = split_chunks(segments)
    conver_to_mp3(chunks)
