from sklearn_crfsuite import metrics
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd

class CrfBrandDetector:
    def __init__(self):
        self.prep_df = None

    def train_test_split(self, prep_df, test_size=0.2, random_state=123):
        self.prep_df = prep_df
        x_train, x_test, y_train, y_test = train_test_split(
            self.prep_df['features'], self.prep_df['labels'],
            test_size=test_size,
            random_state=random_state
        )
        self.test_ind = x_test.index
        return x_train, x_test, y_train, y_test

    def fit(self, x_train, y_train):
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.05,
            c2=0.05,
            max_iterations=100,
            all_possible_states=True
        )
        self.crf.fit(x_train, y_train)


    def save_model(self, filename='helper_files/new_model.sav'):
        pickle.dump(self.crf, open(filename, 'wb'))


    def predict(self, x):
        """
        Returns dataframe with original title and predicted brand.
        If one brand is detected returns string, if more returns list of
        strings.
        """
        title = [[diction['word.lower()'] for diction in obs] for obs in x]
        ind = [[True if elem == 'BRAND' else False for elem in obs]
               for obs in self.crf.predict(x)]
        preds = [' '.join(np.array(title[i])[ind[i]])
                 for i in range(len(title))]
        df_pred = pd.concat([self.prep_df[self.prep_df.index.isin(self.test_ind)].reindex(self.test_ind).reset_index().origin_title,
                            pd.DataFrame(preds)],
                            axis=1
        )

        df_pred.columns = ['title', 'predicted_brand']
        df_pred['predicted_brand'] = df_pred.apply(
                                    lambda row: row.predicted_brand \
                                    if row.predicted_brand in row.title.lower()\
                                    else row.predicted_brand.split(), axis=1)
        return df_pred


    def print_classification_report(self, x_test, y_test):
        """
        Metric explaining how many single tokens in the brand where labeled
        correctly
        """
        labels = list(self.crf.classes_)
        labels.remove('0')
        y_pred = self.crf.predict(x_test)
        print (metrics.flat_classification_report(
            y_test,
            y_pred,
            labels=labels,
            digits=3
        ))


    def evaluate(self, x_test, y_test):
        """
        Metric explaining how many whole titles where labeled correctly
        """
        y_pred = self.crf.predict(x_test)
        acc = float(list(y_test == y_pred).count(True)) / len(y_pred)
        return acc

