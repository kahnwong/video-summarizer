import math
import os
from typing import List

from loguru import logger as log
from pydub import AudioSegment, silence
from tqdm import tqdm

from config import split_audio

# ----- init -----
os.makedirs(split_audio.OUTPUT_DIR, exist_ok=True)


# ----- functions -----
def create_split_points(audio: AudioSegment) -> List[int]:
    log.info("Creating split points")

    # Detect silence ranges
    silences = silence.detect_silence(
        audio,
        min_silence_len=split_audio.MIN_SILENCE_LEN,
        silence_thresh=split_audio.SILENCE_THRESH,
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
        if len(seg) <= split_audio.MAX_CHUNK_MS:
            final_chunks.append(seg)
        else:
            # further subdivide long segments
            parts = math.ceil(len(seg) / split_audio.MAX_CHUNK_MS)
            for i in range(parts):
                sub = seg[
                    i * split_audio.MAX_CHUNK_MS : (i + 1) * split_audio.MAX_CHUNK_MS
                ]
                final_chunks.append(sub)

    return final_chunks


def conver_to_mp3(chunks: List[AudioSegment]) -> None:
    log.info("Converting to mp3")

    for i, chunk in enumerate(tqdm(chunks)):
        out_path = os.path.join(split_audio.OUTPUT_DIR, f"chunk_{i:03d}.mp3")
        chunk.export(out_path, format="mp3")

    log.info(f"Done! Total chunks: {len(chunks)}")


if __name__ == "__main__":
    log.info("Creating audio segments")
    audio = AudioSegment.from_file(split_audio.INPUT_FILE)

    split_points = create_split_points(audio)
    segments = split_by_silence(audio, split_points)
    chunks = split_chunks(segments)
    conver_to_mp3(chunks)
