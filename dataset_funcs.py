import numpy as np
import sys
import os
import librosa
import aux_funcs as aux
import audio_partitioning as audio_aux
import keras

k_VALUE = 10

def load_test_features(saved_files_path, testing_path):
    if not os.path.isfile(saved_files_path + 'feat_test.npy'):
        X_test = aux.get_test_features(testing_path)
        np.save(saved_files_path + 'feat_test.npy', X_test)
        X_test = np.expand_dims(X_test, axis=2)
    else:
        X_test = np.load(saved_files_path + 'feat_test.npy')
        X_test = np.expand_dims(X_test, axis=2)
    return X_test

def load_test_true_labels(saved_files_path, testing_path):
    if not os.path.isfile(saved_files_path + 'label_test.npy'):
        y_test = aux.fill_y_true(testing_path)
        np.save(saved_files_path + 'label_test.npy', y_test)
        y_test = keras.utils.to_categorical(y_test - 1, num_classes=5)
    else:
        y_test = np.load(saved_files_path + 'label_test.npy')
        y_test = keras.utils.to_categorical(y_test - 1, num_classes=5)
    return y_test

def load_train_feat_and_labels(saved_files_path, training_path):
    if not os.path.isfile(saved_files_path + "feat_train.npy") and not os.path.isfile(saved_files_path + "label_train.npy"):
        features, labels = aux.parse_audio_files(training_path) 
        np.save(saved_files_path + 'feat_train.npy', features) 
        np.save(saved_files_path + 'label_train.npy', labels)
        X_train = np.expand_dims(features, axis=2)    
        y_train = labels.ravel()
        y_train = keras.utils.to_categorical(y_train - 1, num_classes=5)
    elif not os.path.isfile(saved_files_path + "feat_train.npy"):
        print("Error in getting features saved file, aborting")
        sys.exit(0)
    elif not os.path.isfile(saved_files_path + "label_train.npy"):
        print("Error in getting labels saved file, aborting")
        sys.exit(0)
    else:
        labels = np.load(saved_files_path + 'label_train.npy')     #5 labels total
        y_train = labels.ravel()
        y_train = keras.utils.to_categorical(y_train - 1, num_classes=5)
        features = np.load(saved_files_path + 'feat_train.npy')
        X_train = np.expand_dims(features, axis=2)
    #print(len(features))
    #print("features", features)

    return X_train, y_train

def load_data(parent_dir, saved_files_dir, seg_ms):
    parent_path = parent_dir+"/"
    saved_files_path = saved_files_dir+"/"
    new_parent_path = "New_5Classes"
    if not os.path.isdir(saved_files_path):
        os.mkdir(saved_files_path)
    new_dir = new_parent_path + "_" + str(seg_ms)
    if not os.path.isdir(saved_files_path + new_dir):
        saved_files_path = saved_files_path + new_dir + "/"
        os.mkdir(saved_files_path)
    else:
        saved_files_path = saved_files_path + new_dir + "/"

    training_path = new_dir + "/Training"
    testing_path = new_dir + "/Testing"
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
        os.mkdir(training_path)
        os.mkdir(testing_path)
        audio_aux.partitionate_audio(parent_path, seg_ms, training_path, testing_path)
    #Get labels and features
    
    X_train, y_train = load_train_feat_and_labels(saved_files_path, training_path)
    X_test = load_test_features(saved_files_path, testing_path)
    y_test = load_test_true_labels(saved_files_path, testing_path)    

    return X_train, X_test, y_train, y_test
 

def load_data_kFold(parent_dir, saved_files_dir, seg_ms):
    parent_path = parent_dir+"/"
    saved_files_path = saved_files_dir+"/"
    new_parent_path = "New_5Classes"
    if not os.path.isdir(saved_files_path):
        os.mkdir(saved_files_path)
    new_dir = new_parent_path + "_" + str(seg_ms)
    
    training_path = []
    testing_path = []
    

    kFold_path = new_dir + "kFold"
    if not os.path.isdir(saved_files_path + kFold_path):
        saved_files_path = saved_files_path + kFold_path + "/"
        os.mkdir(saved_files_path)
        for k in range(0, k_VALUE):
            if not os.path.isdir(saved_files_path + "kFold" + str(k)):
                os.mkdir(saved_files_path + "kFold" + str(k))
    else:
        saved_files_path = saved_files_path + kFold_path + "/"

    if not os.path.isdir(kFold_path):
        os.mkdir(kFold_path)
        for k in range(0, k_VALUE):
            training_path.append(kFold_path + "/Training" + str(k))
            testing_path.append(kFold_path + "/Testing" + str(k))
            if not os.path.isdir(testing_path[k]):
                os.mkdir(testing_path[k])
            if not os.path.isdir(training_path[k]):
                os.mkdir(training_path[k])
            audio_aux.partitionate_audio_kFold(parent_path, seg_ms, training_path[k], testing_path[k], k)
    else:
        for d in os.listdir(kFold_path)[k_VALUE:]:
            training_path.append(kFold_path + "/" + d)
        for d in os.listdir(kFold_path)[:k_VALUE]:
            testing_path.append(kFold_path + "/" + d)
    #Get labels and features
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for k in range(0, k_VALUE):
        X_tr, y_tr = load_train_feat_and_labels(saved_files_path + "kFold" + str(k) + "/", training_path[k])
        X_train.append(X_tr)
        y_train.append(y_tr)
        X_t = load_test_features(saved_files_path + "kFold" + str(k) + "/", testing_path[k])
        y_t = load_test_true_labels(saved_files_path + "kFold" + str(k) + "/", testing_path[k]) 
        X_test.append(X_t)   
        y_test.append(y_t)

    return X_train, X_test, y_train, y_test
