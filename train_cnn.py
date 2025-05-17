import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten,
    Dense, concatenate
)

# Load data
X_seq_df   = pd.read_csv('X_wo_conf.csv')     # one-hot only
X_conf_df  = pd.read_csv('X_with_conf.csv')   # one-hot + confidence column
y_df       = pd.read_csv('y.csv')             # one-hot labels

# Prepare labels and features
y           = y_df.values
num_classes = y.shape[1]
conf        = X_conf_df['confidence'].values.reshape(-1,1)
X_conf_seq  = X_conf_df.drop(columns=['confidence']).values
X_seq       = X_seq_df.values

# Reshape into (samples, 20 positions, 9 categories)
n_samples   = X_seq.shape[0]
X_seq       = X_seq.reshape(n_samples, 20, 9)
X_conf_seq  = X_conf_seq.reshape(n_samples, 20, 9)

# Train/Test splits
X_tr_seq, X_te_seq, y_tr_seq, y_te_seq = train_test_split(
    X_seq, y, test_size=0.3, random_state=42, stratify=y
)
X_tr_cseq, X_te_cseq, conf_tr, conf_te, y_tr_c, y_te_c = train_test_split(
    X_conf_seq, conf, y, test_size=0.3, random_state=42, stratify=y
)

# Model factory functions
def make_seq_cnn(input_shape, num_classes):
    inp = Input(shape=input_shape)
    x   = Conv1D(32, 3, activation='relu')(inp)
    x   = MaxPooling1D(2)(x)
    x   = Conv1D(64, 3, activation='relu')(x)
    x   = MaxPooling1D(2)(x)
    x   = Flatten()(x)
    x   = Dense(100, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def make_seq_conf_cnn(input_shape, num_classes):
    seq_in = Input(shape=input_shape)
    conf_in= Input(shape=(1,))
    x      = Conv1D(32, 3, activation='relu')(seq_in)
    x      = MaxPooling1D(2)(x)
    x      = Conv1D(64, 3, activation='relu')(x)
    x      = MaxPooling1D(2)(x)
    x      = Flatten()(x)
    x      = Dense(100, activation='relu')(x)
    merged = concatenate([x, conf_in])
    m      = Dense(50, activation='relu')(merged)
    out    = Dense(num_classes, activation='softmax')(m)
    model  = Model([seq_in, conf_in], out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Training & evaluation

# CNN WITHOUT confidence (unweighted)
print("CNN WITHOUT confidence (unweighted)")
seq_model = make_seq_cnn((20,9), num_classes)
seq_model.fit(
    X_tr_seq, y_tr_seq,
    validation_split=0.1,
    epochs=100, batch_size=32,
    verbose=2
)
loss_seq, acc_seq = seq_model.evaluate(X_te_seq, y_te_seq, verbose=0)
print(f"Accuracy: {acc_seq:.4f}")

y_true_seq = np.argmax(y_te_seq, axis=1)
y_pred_seq = np.argmax(seq_model.predict(X_te_seq), axis=1)
print(classification_report(y_true_seq, y_pred_seq))

# CNN WITH confidence as feature (unweighted)
print("CNN WITH confidence as feature (unweighted)")
conf_model = make_seq_conf_cnn((20,9), num_classes)
conf_model.fit(
    [X_tr_cseq, conf_tr], y_tr_c,
    validation_split=0.1,
    epochs=100, batch_size=32,
    verbose=2
)
loss_conf, acc_conf = conf_model.evaluate(
    [X_te_cseq, conf_te], y_te_c, verbose=0
)
print(f"Accuracy: {acc_conf:.4f}")

y_true_conf = np.argmax(y_te_c, axis=1)
y_pred_conf = np.argmax(conf_model.predict([X_te_cseq, conf_te]), axis=1)
print(classification_report(y_true_conf, y_pred_conf))

# CNN WITH sequence + sample weights
print("CNN WITH sequence + sample weights")
seq_w_model = make_seq_cnn((20,9), num_classes)
sample_weights = conf_tr.flatten()
seq_w_model.fit(
    X_tr_seq, y_tr_seq,
    sample_weight=sample_weights,
    validation_split=0.1,
    epochs=100, batch_size=32,
    verbose=2
)
loss_w, acc_w = seq_w_model.evaluate(X_te_seq, y_te_seq, verbose=0)
print(f"Accuracy: {acc_w:.4f}")

y_pred_w = np.argmax(seq_w_model.predict(X_te_seq), axis=1)
print(classification_report(y_true_seq, y_pred_w))
