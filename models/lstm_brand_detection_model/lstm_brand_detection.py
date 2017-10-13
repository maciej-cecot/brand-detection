import numpy as np
import pandas as pd
from nltk import word_tokenize
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from crf_layer import ChainCRF
import warnings
warnings.filterwarnings("ignore")

class LstmBrandDetector:
    def __init__(self):
        self.model = None

    def create_model(self, dropout=0.5, units=150):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units, return_sequences=True),
                                     input_shape=(36, 50)))
        self.model.add(Dropout(dropout))
        self.model.add(Bidirectional(LSTM(units, return_sequences=True)))
        self.model.add(Dropout(dropout))
        self.model.add(TimeDistributed(Dense(3)))
        self.model.add(Dropout(dropout))
        crf = ChainCRF()
        self.model.add(crf)
        self.model.compile(loss=crf.loss, optimizer='nadam',
                           metrics=['categorical_accuracy'])

    def fit(self, train_x, train_y, epochs=5, batch=100):
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch)

    def save(self, filepath):
        self.model.save(filepath)

    def print_summary(self):
        print(self.model.summary())

    def predict(self, test_x, test_df=None):
        token_df = test_df.apply(word_tokenize)
        ind = self.model.predict(test_x, verbose=0).argmax(axis=-1)
        ind = [[z for z in obs if z!=2] for obs in ind]
        ind = [[False if elem == 0 else True for elem in obs] for obs in ind]
        output = [' '.join(np.array(token_df[i])[np.array(ind[i])]) for i in range(len(ind))]
        preds = pd.concat([test_df, pd.DataFrame(output, columns=['predictions'])], axis=1)
        return preds

    def evaluate(self, test_x, test_y):
        y_pred = self.model.predict(test_x, verbose=0).argmax(axis=-1)
        y_test = test_y.argmax(axis=-1)
        acc = [np.array_equal(y_pred[i], y_test[i]) for i in
               range(len(y_pred))].count(True) / len(y_pred)
        return acc