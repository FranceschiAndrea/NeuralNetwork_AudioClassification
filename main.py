import keras
from keras.models import load_model
import dataset_funcs as ds_aux
import LSTM_rnn as RNN
import numpy as np
import os


#SET PARAMS:

saved_files_directory = "saved_files"
original_audio_directory = "5Classes"
audio_segmentation_in_ms = 50
epochs_number = 5
work_batch_size = 4096

############

work_directory = "New_5Classes_" + str(audio_segmentation_in_ms) #Togliereil New successivamente +in: --ds_aux.load_data()	--
work_saved_files_path = saved_files_directory + "/" + work_directory + "/"

X_train, X_test, y_train, y_test = ds_aux.load_data(original_audio_directory, saved_files_directory, audio_segmentation_in_ms)

print("\n\n*****************************************Training*****************************************")
if not os.path.isfile(work_saved_files_path + "model_" + str(epochs_number)):
	model = RNN.lstm_RNN(X_train, y_train)
	try:
		model.fit(X_train, y_train, batch_size = work_batch_size, epochs=epochs_number)
	except KeyboardInterrupt:
		pass
	model.save(work_saved_files_path + "model_" + str(epochs_number))
else:
	model = load_model(work_saved_files_path + "model_" + str(epochs_number))

train_acc = model.evaluate(X_train, y_train, verbose=1)
print("\nTrain Statistics: loss = " + str(train_acc[0]) + ", accuracy = " + str(train_acc[1]))

print("\n\n*****************************************Validating*****************************************\n")
score, accuracy = model.evaluate(X_test, y_test, batch_size=work_batch_size, verbose=1)
print("Automatic Generated Model Test Statistics: " + "loss:  " + str(score) + ", accuracy:  " + str(accuracy) + "\n")

#yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=1)
y_true = []
for l in y_test:
    y_true.append(np.argmax(l))
acc = 0
for i in range(0, len(y_true)):
    #print(y_true[i],yhat_classes[i])
    if y_true[i] == yhat_classes[i]:
        acc = acc + 1

res = float(acc/len(y_true))

print("Manually Calculated Model Test Statistics: accuracy: " + str(res) + "\n")

