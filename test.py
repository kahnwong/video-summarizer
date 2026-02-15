import nemo.collections.asr as nemo_asr
from config import asr
import librosa
import soundfile as sf
import tempfile
import os

asr_model = nemo_asr.models.ASRModel.from_pretrained(asr.ASR_MODEL_NAME)

# Load audio and convert to mono if needed
audio_file = "data/source/sample.wav"
audio, sr = librosa.load(audio_file, sr=16000, mono=True)

# Save to temporary file
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp_path = tmp.name
    sf.write(tmp_path, audio, sr)

try:
    transcript = asr_model.transcribe([tmp_path])
    print(f"Transcription: {transcript[0]}")
finally:
    os.unlink(tmp_path)
