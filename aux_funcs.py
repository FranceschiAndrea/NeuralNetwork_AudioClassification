import os
import librosa
import numpy as np
import keras

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sn
import pandas as pd

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
	classes=[]
	for p in os.listdir(testing_path):
		classes.append(p)
		if os.path.isdir(testing_path + '/' + p):
			for a in os.listdir(testing_path + '/' + p):
				y_true.append(i)
		i = i + 1
	y_true = np.array(y_true)
	return y_true


def get_classes(testing_path):
	classes={}
	i = 0
	for p in os.listdir(testing_path):
		classes[p] = i
		i = i + 1
	return classes


def get_evaluation_metrics(model, X_test, y_test, testing_path):
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
	    cnf_mat[inv_map[yhat_classes[i]], inv_map[y_true[i]]] += 1	

	confusion_matrix_array = []
	for c1 in list(classes.keys()):
		l = []
		for c2 in list(classes.keys()):
			l.append(cnf_mat[c1,c2])
		confusion_matrix_array.append(l)

	print(classification_report(y_true, yhat_classes, list(set(y_true))))				#this report show the precision and the recal of each class



		
	df_cm = pd.DataFrame(confusion_matrix_array, index = list(classes.keys()), columns = list(classes.keys()))
	plt.figure(figsize = (10,10))
	plt.title("Confusion Matrix", fontsize=21)
	sn.set(font_scale=1)																
	sn.heatmap(df_cm, fmt = "d", annot=True,annot_kws={"size": 12})						
	tick_marks = np.arange(len(list(classes.keys())))
	plt.xticks(tick_marks, list(classes.keys()), fontsize=12, rotation = 45)
	plt.yticks(tick_marks, list(classes.keys()), fontsize=12, rotation = 30)
	plt.ylabel('True label', fontsize=18)
	plt.xlabel('Predicted label', fontsize=18)
	plt.tight_layout()
	plt.show()

	metrics = ["precision", "recall", "f1-score", "support"]
	metrics_val = []
	cl_rep = classification_report(y_true, yhat_classes, list(set(y_true)), output_dict = True)
	for k in cl_rep:
		print(k)
		d = cl_rep[k]
		l = []
		if (k == "micro avg"):
			break
		for metric in d:
			l.append(d[metric])
		metrics_val.append(l)

	df_cm = pd.DataFrame(metrics_val, index = list(classes.keys()), columns = metrics)
	#print(df_cm)

	plt.figure(figsize = (8,6))
	plt.title("Metrics", fontsize=21)
	sn.set(font_scale=1)																
	sn.heatmap(df_cm, fmt = "f", annot=True,annot_kws={"size": 12}, cmap=ListedColormap(['black']), cbar=False)						
	x_tick_marks = np.arange(len(metrics))
	y_tick_marks = np.arange(len(list(classes.keys())))
	plt.xticks(x_tick_marks, metrics, fontsize=10, rotation = 0)
	plt.yticks(y_tick_marks, list(classes.keys()), fontsize=10)
	plt.ylabel('Classes', fontsize=14)
	plt.xlabel('Metrics', fontsize=14)
	plt.tight_layout()
	plt.show()

	res = float(acc/len(y_true))
	print("Manually Calculated Model Test Statistics: accuracy: " + str(res) + "\n")
