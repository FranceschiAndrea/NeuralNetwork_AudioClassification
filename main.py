import keras
import dataset_funcs as ds_aux
import LSTM_rnn as RNN








X_train, X_test, y_train, y_test = ds_aux.load_data()

model = RNN.lstm_RNN(X_train, y_train)

print("Training ...")
model.fit(X_train, y_train, batch_size = 512, epochs=50)
train_acc = model.evaluate(X_train, y_train, verbose=1)
print("loss, accuracy",train_acc)


print("\nValidating ...")
score, accuracy = model.evaluate(X_test, y_test, batch_size=256, verbose=1)
print("Loss:  ", score)
print("Accuracy:  ", accuracy)





