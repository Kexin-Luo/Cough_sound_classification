import argparse
import torch
import torchvision
import torchaudio
import numpy as np
from PIL import Image
import librosa
import pandas as pd
import os
import pickle as pkl
train_dir="../dataset/csc/audio_train"
val_dir="../dataset/csc/audio_val"
train_meta_dir = "../dataset/csc/meta/CSC4_train.csv"
val_meta_dir = "../dataset/csc/meta/CSC4_val.csv"
store_dir="../dataset/csc/store/"
sr=44100
import matplotlib.pyplot as plt
from librosa import display
audio_dir="../audio_train"
# audio_name="1-2-19.wav"
# audio_path=os.path.join(audio_dir,audio_name)

def extract_spectorgram(values,clip,entries):
    for data in entries:
        num_channels=3
        window_size=[25,50,100]
        hop_size=[10,25,50]
        centre_sec=2.5

        specs=[]
        for i in range(num_channels):
            window_length=int(round(window_size[i]*sr/1000))
            hop_length=int(round(hop_size[i]*sr/1000))
            clip=torch.Tensor(clip)
            spec=torchaudio.transforms.MelSpectrogram(sample_rate=sr,n_fft=4410,win_length=window_length,
                                                      hop_length=hop_length,n_mels=128)(clip)
            eps=1e-6
            spec=spec.numpy()
            spec=np.log(spec+eps)
            spec=np.asarray(torchvision.transforms.Resize((128,250))(Image.fromarray(spec)))
            # print(spec.shape)
            specs.append(spec)

        new_entry={}
        new_entry["audio"]=clip.numpy()
        new_entry["values"]=np.array(specs)

        new_entry["target"]=data["target"]
        values.append(new_entry)

def extract_features(audios,data_dir):
    audio_names=list(audios.filename.unique())
    values=[]
    for audio in audio_names:
        clip,sr=librosa.load("{}/{}".format(data_dir,audio),sr=44100)

        # print("finish audio{}".format(audio))
    return values
def predict_features(audio,data_dir):
        clip,sr=librosa.load("{}/{}".format(data_dir,audio),sr=44100)
        num_channels = 3
        window_size = [25, 50, 100]
        hop_size = [10, 25, 50]
        centre_sec = 2.5

        specs = []
        for i in range(num_channels):
            window_length = int(round(window_size[i] * sr / 1000))
            hop_length = int(round(hop_size[i] * sr / 1000))
            clip = torch.Tensor(clip)
            spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=4410, win_length=window_length,
                                                        hop_length=hop_length, n_mels=128)(clip)
            eps = 1e-6
            spec = spec.numpy()
            spec = np.log(spec + eps)
            spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
            # print(spec.shape)
            specs.append(spec)
        return specs
if __name__ == '__main__':

    
    train_data=pd.read_csv(train_meta_dir,skipinitialspace=True)
#     val_data=pd.read_csv(val_meta_dir,skipinitialspace=True)
#     audio_path=os.path.join(audio_dir,audio_name)
    training_values=extract_features(train_data,audio_dir)
    
#     librosa.display.specshow(training_values, sr=sr, x_axis='time', y_axis='mel')
#     plt.ylabel('Mel Frequency')
#     plt.xlabel('Time(s)')
#     plt.title('Mel Spectrogram')
#     plt.savefig(fname="melsp.png",figsize=[10,10])
#     plt.show()
#     training_values=extract_features(train_data,train_dir)
#     print(np.array(training_values).shape)
#     with open("{}training128mel.pkl".format(store_dir),"wb")as handler:
#         pkl.dump(training_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

#     validation_values = extract_features(val_data,val_dir)
#     with open("{}validation128mel.pkl".format(store_dir), "wb") as handler:
#         pkl.dump(validation_values, handler, protocol=pkl.HIGHEST_PROTOCOL)


