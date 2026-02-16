# Video Summarizer

Refactored version of <https://github.com/ro-witthawin/videoSummarizationWithASRandVisual>.

## Data

```bash
mkdir -p data/source
yt-dlp -f 140+137 https://www.youtube.com/watch?v=oTGOpu2eyIc
```

## Usage

```bash
cd data/source
ffmpeg -i video.mp4 audio.wav
```
