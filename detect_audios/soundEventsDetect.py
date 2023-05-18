from pydub import AudioSegment
import wave
import auditok
import matplotlib.pyplot as plot
from scipy.io import wavfile
import os
import torch
from preprocessing import preprocessingCSC
import numpy as np
import pyaudio
import audio_monitor
import noisereduce
# name="../croup_test.wav"
def genCoughEvents(name):
    region = auditok.load(name)
    audio_regions = region.split(
        min_dur=1,        # 声音事件的最短长度
        max_dur=5,          # 声音事件的最长长度
        max_silence=0.3,    # 声音事件中无信号最长长度
        energy_threshold=40, # 侦测声音事件的能量门槛值
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
if __name__ == '__main__':
    name="../whooping_2.wav"
    audio_monitor.get_audio(name)
    genCoughEvents(name)