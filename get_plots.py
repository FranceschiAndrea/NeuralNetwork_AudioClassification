import librosa
from librosa import display
import matplotlib.pyplot as plt
import os
import sklearn
from matplotlib.colors import ListedColormap
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def get_classes(testing_path):
	classes={}
	i = 0
	for p in os.listdir(testing_path):
		classes[p] = i
		i = i + 1
	return classes

def get_spectogram(x,sr, work_saved_files_path):
	#display Spectrogram
	'''
	for d in os.listdir(audio_path):
		for file in os.listdir(audio_path + "/" + d):
			x,sr = librosa.load(audio_path + "/" + d + "/" + file)
			stft = librosa.stft(x)
			stft_db = librosa.amplitude_to_db(abs(stft))
			plt.figure(figsize=(14, 5))
			librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz') 
			plt.colorbar()
			plt.show()
			#plt.savefig(work_saved_files_path + file + "_spectogram", bbox_inches='tight')
			break
		break
	'''

	#display Spectrogram of first audio

	stft = librosa.stft(x)
	stft_db = librosa.amplitude_to_db(abs(stft))
	plt.figure(figsize=(14, 5))
	librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz') 
	plt.colorbar()
	plt.savefig(work_saved_files_path, bbox_inches='tight')
	plt.show()

	#display Spectrogram of first audio segment
	'''
	d = os.listdir(testing_path)[0]
	audio_dir = os.listdir(testing_path + "/" +d)[0]
	x,sr = librosa.load(testing_path + "/" + d + "/" + audio_dir)
	stft = librosa.stft(x)
	stft_db = librosa.amplitude_to_db(abs(stft))
	plt.figure(figsize=(14, 5))
	librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz') 
	plt.colorbar()
	plt.show()
	'''


def get_spectral_centroid(x,sr, work_saved_files_path):
	#spectral centroid  
	#centre of mass  
	#weighted mean of the frequencies present in the sound

	spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
	spectral_centroids.shape
	frames = range(len(spectral_centroids))
	t = librosa.frames_to_time(frames)
	def normalize(x, axis=0):
	    return sklearn.preprocessing.minmax_scale(x, axis=axis)

	#Plotting the Spectral Centroid along the waveform
	librosa.display.waveplot(x, sr=sr, alpha=0.4)
	plt.plot(t, normalize(spectral_centroids), color='r')
	plt.savefig(work_saved_files_path, bbox_inches='tight')
	plt.show()

def get_mfcc(x, sr, work_saved_files_path):
	#get Mel-Frequency Cepstral Coefficients
	mfccs = librosa.feature.mfcc(x, sr=sr)
	#print(mfccs.shape)
	librosa.display.specshow(mfccs, sr=sr, x_axis='time')
	plt.savefig(work_saved_files_path, bbox_inches='tight')
	plt.show()


def create_plots(audio_path, work_saved_files_path, testing_path):
	d = os.listdir(audio_path)[0]	
	file = os.listdir(audio_path + "/" + d)[0]
	x,sr = librosa.load(audio_path + "/" + d + "/" + file)		#first audio segment
	get_spectogram(x,sr,work_saved_files_path + file.split(".")[0] + "_spectogram.png")
	get_spectral_centroid(x,sr,work_saved_files_path + file.split(".")[0] + "_spectral_centroid.png")
	get_mfcc(x,sr,work_saved_files_path + file.split(".")[0] + "_mfcc.png")


def get_model_plots(history, work_saved_files_path):
	if history:
		plt.title('Model Loss')
		plt.plot(history.history['loss'], label='train')
		plt.savefig(work_saved_files_path + "model_loss", bbox_inches='tight')
		plt.show()

		plt.title('Model Accuracy')
		plt.plot(history.history['acc'], label='train')
		plt.savefig(work_saved_files_path + "model_accuracy", bbox_inches='tight')
		plt.show()





def get_evaluation_metrics(model, X_test, y_test, testing_path, work_saved_files_path):
	classes = get_classes(testing_path)
	cnf_mat = {}												#confusion matrix
	for c1 in list(classes.keys()):
		for c2 in list(classes.keys()):
			cnf_mat[(c1,c2)] = 0

	inv_map = {v: k for k, v in classes.items()}

	yhat_classes = model.predict_classes(X_test, verbose=1)
	y_true = []
	for l in y_test:
	    y_true.append(np.argmax(l))
	acc = 0
	for i in range(0, len(y_true)):
	    if y_true[i] == yhat_classes[i]:
	        acc = acc + 1

	res = float(acc/len(y_true))
	print("Manually Calculated Model Test Statistics: accuracy: " + str(res) + "\n-------------------------------------------------")

	return res