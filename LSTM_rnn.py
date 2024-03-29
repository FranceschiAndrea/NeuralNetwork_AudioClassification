import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam

# Build RNN Neural Network
def lstm_RNN(X_train, y_train):
    print('Build LSTM RNN model ...')
    model = Sequential()
    model.add(keras.layers.CuDNNLSTM(128, return_sequences=True, input_shape=X_train.shape[1:]))
    model.add(keras.layers.CuDNNLSTM(32, return_sequences=False))
    #model.add(Dense(y_train.shape[1], activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    print("Compiling ...")
    model.compile(loss='categorical_crossentropy', # for multiple classes
                  optimizer='adam', 
                  metrics=['accuracy'])
    


    print(model.summary())
    return model

'''
model.add(LSTM(64, return_sequences=False, 
               dropout=0.1, recurrent_dropout=0.1))

# Fully connected layer
model.add(Dense(64, activation='relu'))

# Dropout for regularization
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_words, activation='softmax'))
'''