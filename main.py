import keras
from keras.models import load_model
import dataset_funcs as ds_aux
import LSTM_rnn as RNN
import numpy as np
import os




saved_files_path = "saved_files/"
parent_path = "5Classes/"
new_parent_path = "New_5Classes"
seg_ms = 50
new_dir = new_parent_path + "_" + str(seg_ms)
saved_files_path = saved_files_path + new_dir + "/"

epochs = 100

X_train, X_test, y_train, y_test = ds_aux.load_data()

print("Training ...")
if not os.path.isfile(saved_files_path + "model_" + str(epochs)):
	model = RNN.lstm_RNN(X_train, y_train)
	try:
		model.fit(X_train, y_train, batch_size = 1024, epochs=epochs)
	except KeyboardInterrupt:
		pass
	model.save(saved_files_path + "model_" + str(epochs))
else:
	model = load_model(saved_files_path + "model_" + str(epochs))




train_acc = model.evaluate(X_train, y_train, verbose=1)
print("loss, accuracy",train_acc)


print("\nValidating ...")
score, accuracy = model.evaluate(X_test, y_test, batch_size=1024, verbose=1)
print("Loss:  ", score)
print("Accuracy:  ", accuracy)



#yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=1)
y_true = []
for l in y_test:
    y_true.append(np.argmax(l))
acc = 0
for i in range(0, len(y_true)):
    print(y_true[i],yhat_classes[i])
    if y_true[i] == yhat_classes[i]:
        acc = acc + 1

res = float(acc/len(y_true))
print("L'accuracy è : ", res)

