import string
from nltk.tokenize import word_tokenize
import pandas as pd


def labeling(label, string_chunk):
    return [(word, label) for word in word_tokenize(string_chunk)]

def branding(df):
    t, br = df.title, df.brand
    start = (t.lower()).find(br.lower())
    end = start + len(br)
    labeled_title = labeling('0', t[:start]) \
                    + labeling('BRAND', t[start:end]) + labeling('0', t[end:])
    return labeled_title

def word2features(sent, i):
    word = str(sent.title[i][0])
    features = {
        'root_cat': sent.root_cat,
        'bias': 1,
        'word_position': i,
        'word.lower()': word.lower(),
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }

    if i > 0:
        word1 = str(sent.title[i - 1][0])
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })

    else:
        features['BOS'] = True

    if i < len(sent.title) - 1:
        word1 = str(sent.title[i + 1][0])
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
        word1 = str(sent.title[i - 1][0])
        word2 = str(sent.title[i - 2][0])
        features.update({
            '-2:ngram': '{} {}'.format(word2, word1)
        })

    if i < len(sent.title) - 2:
        word1 = str(sent.title[i + 1][0])
        word2 = str(sent.title[i + 2][0])
        features.update({
            '+2:ngram': '{} {}'.format(word1, word2)
        })

    return features


class DataPreparation:
    """
    Required data format:
    df.title - title of ebay product
    df.brand - brand attribute of ebay product
    df.root_cat - root category of item
    """

    def features_labels_prep(self, filename='helper_files/train.csv'):
        dfr = pd.read_csv(filename)
        dfr['origin_title'] = dfr['title'].values
        dfr['title'] = dfr.apply(branding, axis=1)
        dfr['features'] = dfr.apply(lambda row:
                                    [word2features(row, i)
                                    for i in range(len(row.title))],
                                    axis=1)
        dfr['labels'] = dfr.apply(lambda row:
                                  [label for token, label in row.title],
                                  axis=1)

        return dfr