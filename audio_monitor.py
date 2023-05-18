from pydub import AudioSegment
from pydub.effects import normalize
from pydub.playback import play
import wave
import auditok
import matplotlib.pyplot as plot
from scipy.io import wavfile
import os
import torch
from preprocessing import preprocessingCSC
import numpy as np
import pyaudio
import time
from detect_audios import soundEventsDetect
# 读取mp3的波形数据


def getaudioname():
    dataname=os.listdir(audio_dir)
    list=[]
    for i in dataname:
        list.append(i)
    list=list[0:-2]
    return list


def get_audio(filepath):
    CHUNK = 256
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 声道数
    RATE = 44100  # 采样率
    RECORD_SECONDS = 6
    WAVE_OUTPUT_FILENAME = filepath
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("*" * 6, "开始录音：请在60秒内输入语音")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("*" * 6, "录音结束\n")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
def detector():
    region = auditok.load(name)
    audio_regions = region.split(
        min_dur=1,        # 声音事件的最短长度
        max_dur=5,          # 声音事件的最长长度
        max_silence=0.3,    # 声音事件中无信号最长长度
        energy_threshold=55, # 侦测声音事件的能量门槛值
        channels=1
    )
    for i, r in enumerate(audio_regions):

        #输出每段分割音频的起始与结束时间点
        print("Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))
        #拨放每段分割音频
        # r.play(progress_bar=True)
        #保存每段分割音频
        filename = r.save("{meta.start:.3f}.wav")
        # audio_regions.close()
        # sound1 = AudioSegment.from_wav(name)
        # new_sound = sound1 + silent
        # new_sound.export(name, format="wav")
        # music = wavfile.read(name)
        # wavfile.write(name, 44100, music[1][0 * 44100:5 * 44100])
        # data = preprocessingCSC.predict_feature(name)
        # data=np.array(data)
        # data=np.expand_dims(data,axis=0)
        # data=torch.tensor(data)
        # pred = model(data)
        # pred=pred.cpu().detach().numpy()
        # res = np.argmax(pred,axis=1)
        # print(res)
        # # print("保存为：{}".format(filename))
def classification(namelist,model_path):
    for audio in namelist:
        data = preprocessingCSC.predict_features(audio, audio_dir)
        data = np.array(data)
        data = np.expand_dims(data, axis=0)
        data = torch.tensor(data)
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        begin = time.clock()
        pred = model(data)
        end = time.clock()
        print(end - begin)
        pred = pred.cpu().detach().numpy()
        res = np.argmax(pred, axis=1)
        print(audio)
        if res==0:
            print("other cough")
        if res==1:
            print("pertussis")
        if res==2:
            print("croup")
        if res==3:
            print("noncough")
if __name__ == '__main__':
    silent = AudioSegment.silent(duration=5000)
    name = 'croup.wav'
    loud_quiet = AudioSegment.from_file(name)
    # Normalize the sound levels
    normalized_loud_quiet = normalize(loud_quiet)
    # Check the sound
    play(normalized_loud_quiet)
    # region.plot()
    # model_path = "model/mel_senet_3.pkl"
    # model_path_inception = "model/mel_inception_3.pkl"
    # model_path_lstm = "model/mel_lstm_3.pkl"
    # model_path_melcnn = "model/mel_melcnn_3.pkl"
    # model_path_mobilenet = "model/mel_mobilenet_3.pkl"
    # model_path_resnet = "model/mel_resnet_3.pkl"
    # model_path_resnext = "model/mel_resnext_3.pkl"
    # model_path_senet = "model/mel_senet_3.pkl"
    # audio_dir = "detect_audios"
    # namelist=getaudioname()
    # classification(namelist,model_path_lstm)

    # end.record()
    # torch.cuda.synchronize()
    # print(start.elapsed_time(end))
