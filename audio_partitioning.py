from pydub import AudioSegment
import os, sys, stat
import librosa
import numpy as np
import random

def partitionate_audio(directory, seg_ms, training_path, testing_path):
	audio_directories = os.listdir(directory)
	audio_directories.sort()
	#2 params to set
	simple_rate = 22050
	frame_size_insec = seg_ms/1000
	#
	frame_size = int(round(simple_rate*frame_size_insec))
	hop = int(round(simple_rate*frame_size_insec))
	for d in audio_directories:
		path_directories = directory+d
		for f in os.listdir(path_directories):
			path_wav_file = path_directories+"/"+f
			#j = 0
			print("Loading...: "+path_wav_file)
			audio, sr = librosa.load(path_wav_file, sr=simple_rate)
			partitioned_audio = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop)			
			frames_number = partitioned_audio.shape[1]
			print("Partitioning...: "+d+" in "+str(frames_number)+" frames of "+str(frame_size_insec)+" seconds")
			train_segm_len = int(0.75*frames_number)
			test_segm_len = int(0.25*frames_number)
			l=list(range(frames_number))
			random.shuffle(l)
			l_train=l[:train_segm_len]
			l_test=l[train_segm_len:]
			for i in l_train:
				frame = partitioned_audio[:,i]
				path_new_folder = training_path+"/"+d
				path_new_audio = path_new_folder+"/"+str(f).split(".")[0] + "_" + str(i)+".wav"
				if not os.path.isdir(path_new_folder):
					os.mkdir(path_new_folder)
					os.chmod(path_new_folder, 0o777)
				librosa.output.write_wav(path_new_audio, frame, sr)	
			for i in l_test:
				frame = partitioned_audio[:,i]
				path_new_folder = testing_path+"/"+d
				path_new_audio = path_new_folder+"/"+str(f).split(".")[0] + "_" + str(i)+".wav"
				if not os.path.isdir(path_new_folder):
					os.mkdir(path_new_folder)
					os.chmod(path_new_folder, 0o777)
				librosa.output.write_wav(path_new_audio, frame, sr)	
				#j = j + 1

