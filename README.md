# Video Summarizer

Refactored version of <https://github.com/ro-witthawin/videoSummarizationWithASRandVisual>.

## Data

```bash
mkdir -p data/source
yt-dlp -f 140+137 https://www.youtube.com/watch?v=oTGOpu2eyIc
```

## Prep

```bash
cd data/source
ffmpeg -i video.mp4 audio.wav
```

## Usage

Run scripts in ascending order.

## Bugs

For final summarization, sometimes the return timestamp is in `ms`, sometimes `s`.
