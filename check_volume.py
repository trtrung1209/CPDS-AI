import librosa
import numpy as np
from pathlib import Path

def get_avg_rms(directory):
    rms_list = []
    files = list(Path(directory).rglob("*.wav"))
    for f in files[:100]: # Sample 100
        y, _ = librosa.load(f, sr=16000)
        rms = np.sqrt(np.mean(y**2))
        rms_list.append(rms)
    return np.mean(rms_list)

cry_dir = "/media/trtrung1209/WORKSPACES/WORKSPACES/01_PROJECTS/2026_NCKH_CPDS-AI/data/test_audio/cry"
noise_dir = "/media/trtrung1209/WORKSPACES/WORKSPACES/01_PROJECTS/2026_NCKH_CPDS-AI/data/test_audio/noise"

print(f"Average RMS of Cry: {get_avg_rms(cry_dir):.5f}")
print(f"Average RMS of Noise: {get_avg_rms(noise_dir):.5f}")
