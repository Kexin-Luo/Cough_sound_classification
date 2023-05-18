import librosa
import argparse
import pandas as pd
import numpy as np
import pickle as pkl 
import torch
import torchaudio
import torchvision
from PIL import Image

parser = argparse.ArgumentParser()
# parser.add_argument("--csv_file", type=str)
# parser.add_argument("--data_dir", type=str)
# parser.add_argument("--store_dir", type=str)
# parser.add_argument("--sampling_rate", default=44100, type=int)
# audio_dir="../dataset/esc50/audio"
# # val_dir="../dataset/audio_val"
# audio_meta_dir = "../dataset/esc50/meta/esc50.csv"
# # val_meta_dir = "../dataset/meta/CSC4_val.csv"
# store_dir="../dataset/esc50/store"
audio_dir="../dataset/data/audio"
audio_meta_dir = "../dataset/data/meta/Coswara.csv"
store_dir="../dataset/data/store"
sampling_rate=22050
def extract_spectrogram(values, clip, entries):
	for data in entries:

		num_channels = 3
		window_sizes = [25, 50, 100]
		hop_sizes = [10, 25, 50]
		centre_sec = 2.5

		specs = []
		for i in range(num_channels):
			window_length = int(round(window_sizes[i]*sampling_rate/1000))
			hop_length = int(round(hop_sizes[i]*sampling_rate/1000))

			clip = torch.Tensor(clip)
			spec = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_fft=4410, win_length=window_length, hop_length=hop_length, n_mels=128)(clip)
			eps = 1e-6
			spec = spec.numpy()
			spec = np.log(spec+ eps)
			spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
			specs.append(spec)
		new_entry = {}
		new_entry["audio"] = clip.numpy()
		new_entry["values"] = np.array(specs)
		new_entry["target"] = data["target"]
		values.append(new_entry)

def extract_features(audios,data_dir):
	audio_names = list(audios.filename.unique())
	values = []
	for audio in audio_names:
		clip, sr = librosa.load("{}/{}".format(data_dir, audio), sr=sampling_rate)
		entries = audios.loc[audios["filename"]==audio].to_dict(orient="records")
		extract_spectrogram(values, clip, entries)
		print("Finished audio {}".format(audio))
	return values

if __name__=="__main__":
	args = parser.parse_args()
	audios = pd.read_csv(audio_meta_dir, skipinitialspace=True)
	num_folds = 2
	training_audios = audios.loc[audios["fold"]==1]
	validation_audios = audios.loc[audios["fold"]==2]
	test_audios = audios.loc[audios["fold"] == 3]

	# training_values = extract_features(training_audios,audio_dir)
	# with open("{}training128mel{}.pkl".format(store_dir, 1),"wb") as handler:
	# 	pkl.dump(training_values, handler, protocol=pkl.HIGHEST_PROTOCOL)
	#
	# validation_values = extract_features(validation_audios,audio_dir)
	# with open("{}validation128mel{}.pkl".format(store_dir, 1),"wb") as handler:
	# 		pkl.dump(validation_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

	test_values = extract_features(test_audios, audio_dir)
	with open("{}test128mel{}.pkl".format(store_dir, 1), "wb") as handler:
		pkl.dump(test_values, handler, protocol=pkl.HIGHEST_PROTOCOL)