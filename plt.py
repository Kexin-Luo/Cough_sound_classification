import librosa, librosa.display
import matplotlib.pyplot as plt

file = 'noise audio/cough_music_reduce.wav'
signal, sr = librosa.load(file, sr=44100)
librosa.display.waveshow(signal, sr=sr)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.savefig(fname="music_cough_reduce.png",figsize=[10,10])
plt.show()

