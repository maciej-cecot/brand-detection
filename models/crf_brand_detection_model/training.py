from data_preparation import DataPreparation
from crf_brand_detection import CrfBrandDetector

if __name__ == "__main__":
    print('Data preparing..')
    prep_df = DataPreparation().features_labels_prep()
    print('Model fitting...')
    model = CrfBrandDetector()
    x_train, x_test, y_train, y_test = model.train_test_split(prep_df)
    model.fit(x_train, y_train)
    model.print_classification_report(x_test, y_test)
    print('Accuracy for whole titles: {}'.format(model.evaluate(x_test, y_test)))
    pred = model.predict(x_test)
    pred.to_csv('helper_files/predictions.csv')