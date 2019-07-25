import keras
from keras.models import load_model
import dataset_funcs as ds_aux
import LSTM_rnn as RNN
import numpy as np
import os
import aux_funcs as aux
import get_plots as plt
#SET PARAMS:

saved_files_directory = "saved_files"
original_audio_directory = "5Classes"
audio_segmentation_in_ms = 60000
epochs_number = 20
work_batch_size = 1024
work_directory = "New_5Classes_" + str(audio_segmentation_in_ms) + "kFold" #Togliereil New successivamente +in: --ds_aux.load_data()	--
work_saved_files_path = saved_files_directory + "/" + work_directory + "/"
training_path = work_directory + "/Training"
testing_path = work_directory + "/Testing"

history = None

k_VALUE = 10
############



X_train, X_test, y_train, y_test = ds_aux.load_data_kFold(original_audio_directory, saved_files_directory, audio_segmentation_in_ms)

res = []

print("\n\n*****************************************Training*****************************************")
for k in range(0, k_VALUE):
	if not os.path.isfile(work_saved_files_path + "model_" + str(epochs_number)):
		model = RNN.lstm_RNN(X_train[k], y_train[k])
		try:
			history = model.fit(X_train[k], y_train[k], batch_size = work_batch_size, epochs=epochs_number)
		except KeyboardInterrupt:
			pass
		model.save(work_saved_files_path + "model_" + str(epochs_number))
	else:
		model = load_model(work_saved_files_path + "model_" + str(epochs_number))

	train_acc = model.evaluate(X_train[k], y_train[k], verbose=1)
	print("\nTrain Statistics: loss = " + str(train_acc[0]) + ", accuracy = " + str(train_acc[1]))


	print("\n\n*****************************************Validating*****************************************\n")
	score, accuracy = model.evaluate(X_test[k], y_test[k], batch_size=work_batch_size, verbose=1)
	print("Automatic Generated Model Test Statistics: " + "loss:  " + str(score) + ", accuracy:  " + str(accuracy) + "\n")

	res.append(plt.get_evaluation_metrics_kFold(model, X_test[k], y_test[k], testing_path + str(k), work_saved_files_path))


print("Final accuracy: ", sum(res)/10)



