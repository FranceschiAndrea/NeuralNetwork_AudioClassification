import os
import librosa
import numpy as np
import keras

def feature_extraction(file_name):
	audio, sample_rate = librosa.load(file_name)    #load the audio file in wav format
	if audio.ndim > 1:
		audio = audio[:,0]
	audio = audio.T

    # Get features   
	stft = np.abs(librosa.stft(audio))
	mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0) #40 values
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
	mel = np.mean(librosa.feature.melspectrogram(audio, sr=sample_rate).T, axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
	#tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T, axis=0) #tonal centroid features

    #Return computed features
	return mfccs, chroma, mel, contrast
    
# Process audio files: Return arrays with features and labels
def parse_audio_files(train_path): 
	features, labels = np.empty((0,187)), np.empty(0) #193 features total for each file
	classes = {}
	i = 0
	for d in os.listdir(train_path):        #create a dict with key the dir. name, value an integer label index
		classes[d] = i
		i = i + 1
	for d in os.listdir(train_path):        
		i = 0
		label = classes[d]
		print(d, classes[d])
		for file_name in os.listdir(train_path + "/" + d + "/"): 
			#a = librosa.load(train_path + "/" + d + "/" + file_name)
			#print(a)
			try:
				mfccs, chroma, mel, contrast = feature_extraction(train_path + "/" + d + "/" + file_name)          #extract feature for each file
			except Exception as e:
				print("[Error] there was an error in feature extraction. %s" % (e))
				continue
			extracted_features = np.hstack([mfccs,chroma, mel, contrast])                 # stack arrays in sequence horizontally (column wise)
			features = np.vstack([features, extracted_features])    # stack arrays in sequence vertically (row wise).
			labels = np.append(labels, label)
			i = i + 1
			if i % 100 == 0:
				print("Extracted features from: %d files" % (i))
		print("Extracted features from %s, done" % (d))
	return np.array(features), np.array(labels, dtype = np.int)     # arrays with features and corresponding labels for each audio


def get_test_features(testing_path):
	features = np.empty((0,187))
	for p in os.listdir(testing_path):
		if os.path.isdir(testing_path + '/' + p):
			for a in os.listdir(testing_path + '/' + p):
				audio, sample_rate = librosa.load(testing_path + '/' + p + '/' + a)
				stft = np.abs(librosa.stft(audio))
				mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0) #40 values
				chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
				mel = np.mean(librosa.feature.melspectrogram(audio, sr=sample_rate).T, axis=0)
				contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
				#tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T, axis=0) #tonal centroid features
				extracted_features = np.hstack([mfccs,chroma, mel, contrast]) 
				features = np.vstack([features, extracted_features]) 
	features = np.array(features)
	return features



def fill_y_true(testing_path):
	y_true = []
	i = 0
	for p in os.listdir(testing_path):
		if os.path.isdir(testing_path + '/' + p):
			for a in os.listdir(testing_path + '/' + p):
				y_true.append(i)
		i = i + 1
	y_true = np.array(y_true)
	return y_true


