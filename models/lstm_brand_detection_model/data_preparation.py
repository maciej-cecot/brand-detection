import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from itertools import chain


def labeling(label, string_chunk):
    return [(word, label) for word in word_tokenize(string_chunk)]

def branding(df):
    t, br = df.title, df.brand
    start = (t.lower()).find(br.lower())
    end = start + len(br)
    labeled_title = labeling('0', t[:start]) + \
                    labeling('BRAND', t[start:end]) + labeling('0', t[end:])
    return labeled_title

def one_hot_encoding(x, y):
    y = np.array(y)
    max_sentence_len = max(map(len, x))
    all_tags = set(chain(*y))
    NUM_TAGS = len(all_tags)
    TAGS_MAP = dict(zip(all_tags, range(NUM_TAGS)))
    TAGS_MAP = {'0': 0, 'BRAND': 1, 'NIL': 2}
    y = list(map(lambda x: [TAGS_MAP[t] for t in x], y))
    y = pad_sequences(y, max_sentence_len, padding='pre')
    y = np.array([to_categorical(t, NUM_TAGS) for t in y])
    return x, y


class DataPreparation:
    def __init__(self, titles_filepath, default_n_classes=3):
        self.default_n_classes = default_n_classes
        self.titles_filepath = titles_filepath

    def load_glove(self, glove_path):
        self.wordvecs = open(glove_path, encoding='utf-8')
        self.word_to_ix_map = {}
        for line in self.wordvecs:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.word_to_ix_map[word] = coefs
        self.wordvecs.close()

    def read_data(self):
        """
        Reads data from the dataframe.
        Columns that have to be included: [title, brand]
        """
        df = pd.read_csv(self.titles_filepath)
        df['origin_title'] = df['title'].values
        df['title'] = df.apply(branding, axis=1)
        self.df = df
        raw_w, raw_t, raw_data, = [], [], []
        for row in self.df.title:
            for word, tag in row:
                raw_w.append(word)
                raw_t.append(tag)
            raw_data.append((tuple(raw_w), tuple(raw_t)))
            raw_w, raw_t = [], []
        self.raw_data = raw_data

    def prepare_data(self):
        all_x, all_y = [], []
        max_sentence_len = 36
        for words, tags in self.raw_data:
            encoded_words, encoded_tags = [], []
            for w, t in zip(words, tags):
                if w.lower() in self.word_to_ix_map:
                    encoded_words.append(self.word_to_ix_map[w.lower()])
                    encoded_tags.append(t)
                else:
                    encoded_words.append(np.ones(50))
                    encoded_tags.append(t)
            nil_x = np.zeros(50)
            nil_y = 'NIL'
            pad_length = max_sentence_len - len(encoded_words)
            all_x.append(((pad_length) * [nil_x]) + encoded_words)
            all_y.append(((pad_length) * [nil_y]) + encoded_tags)
        all_x, all_y = one_hot_encoding(all_x, all_y)
        all_x, all_y = np.array(all_x), np.array(all_y)
        return all_x, all_y

    def train_test_split(self, all_x, all_y, test_size=0.2):
        x_train, x_test, y_train, y_test, ind_train, ind_test = train_test_split(all_x, all_y, range(len(all_x)), test_size=test_size)
        test_df = self.df[self.df.index.isin(ind_test)].reindex(ind_test).reset_index().origin_title
        return x_train, x_test, y_train, y_test, test_df