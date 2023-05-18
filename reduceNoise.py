from scipy.io import wavfile
import noisereduce as nr
import pyaudio
import time
import wave


def reduceNoise(audio,noise):
    rate, data = wavfile.read(audio)
    _,noisy_part =  wavfile.read(noise)
    SAMPLING_FREQUENCY=44100
    reduced_noise = nr.reduce_noise(y=data, y_noise=noisy_part, sr=SAMPLING_FREQUENCY)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = time
    WAVE_OUTPUT_FILENAME = "out_file.wav"

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(reduced_noise))


if __name__ == '__main__':
    audio="noise audio/cough_music.wav"
    music="noise audio/music.wav"
    talking="noise audio/talking.wav"
    mid="out_file.wav"
    reduceNoise(audio,music)
    reduceNoise(mid,talking)