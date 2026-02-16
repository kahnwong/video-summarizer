import glob
import json
import os
from typing import Any, List

import cv2
from loguru import logger as log

from config import create_markdown

# ----- init -----
os.makedirs(create_markdown.FRAMES_DIR, exist_ok=True)


# ----- utils -----
def get_video_filename() -> str:
    files = glob.glob(create_markdown.VIDEO_DIR + "/*.mp4")
    return files[0]


# ----- functions -----
def extract_frames(selected_ts) -> List[Any]:
    log.info("Extracting frames...")
    cap = cv2.VideoCapture(get_video_filename())
    frames = []

    for ts in selected_ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, frame = cap.read()
        if ok:
            fpath = f"{create_markdown.FRAMES_DIR}/frame_{int(ts)}.jpg"
            cv2.imwrite(fpath, frame)
            frames.append((ts, f"frames/frame_{int(ts)}.jpg"))

    return frames


def inject_image_markdown(frames: List[Any], summary_md: str) -> str:
    log.info("Injecting images...")
    output = ""
    for ts, path in frames:
        output = summary_md.replace(f"(frames/frame_{int(ts)}.jpg)", f"({path})")

    return output


if __name__ == "__main__":
    with open(create_markdown.INPUT_FILE) as f:
        summary_raw = json.load(f)

    frames = extract_frames(summary_raw["selected_timestamps"])
    markdown = inject_image_markdown(frames, summary_raw["summary_md"])

    with open(create_markdown.OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(markdown)
