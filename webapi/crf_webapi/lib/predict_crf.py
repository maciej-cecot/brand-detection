import pandas as pd
import numpy as np
import pickle
from nltk import word_tokenize
import string


def word2features(sent, i):
    word = str(sent.title[i][0])
    features = {
            'root_cat': sent.root_cat,
            'bias':1,
            'word_position': i,
            'word.lower()': word.lower(),
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }

    if i > 0:
        word1 = str(sent.title[i-1][0])
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })

    else:
        features['BOS'] = True

    if i < len(sent.title)-1:
        word1 = str(sent.title[i+1][0])
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.anydigit': any(ch.isdigit() for ch in word1),
            '+1:word.ispunctuation': word1 in string.punctuation,
        })

    else:
        features['EOS'] = True

    if i > 1:
        word1 = str(sent.title[i-1][0])
        word2 = str(sent.title[i-2][0])
        features.update({
            '-2:ngram': '{} {}'.format(word2, word1)
        })

    if i < len(sent.title)-2:
        word1 = str(sent.title[i+1][0])
        word2 = str(sent.title[i+2][0])
        features.update({
            '+2:ngram': '{} {}'.format(word1, word2)
        })

    return features

def predict_only(titles, model_filepath):
    try:
        crf = pickle.load(open(model_filepath, 'rb'))
    except:
        raise Exception('Load the model before making predictions')
    x_test = pd.DataFrame(titles, columns=titles.keys())
    x_test['original_title'] = x_test.title.values
    x_test['title'] = x_test['original_title'].apply(word_tokenize)
    x_test['title'] = x_test.title.apply(lambda row: [[x] for x in row])
    x_test['features'] = x_test.apply(
        lambda row: [word2features(row, i) for i in range(len(row.title))],
        axis=1)
    ind = [[True if elem == 'BRAND' else False for elem in obs] for obs in
           crf.predict(x_test.features)]
    preds = [' '.join(np.array(word_tokenize(x_test.iloc[i].original_title))[ind[i]])
             for i in range(len(x_test))]
    return preds




