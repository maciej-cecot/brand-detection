from data_preparation import DataPreparation
from lstm_brand_detection import LstmBrandDetector


if __name__ == "__main__":
    prep = DataPreparation(titles_filepath = 'helper_files/train.csv')
    print('Loading word embeddings...')
    prep.load_glove('helper_files/glove.txt')
    print('Reading data...')
    prep.read_data()
    print('Data preparing..')
    x, y = prep.prepare_data()
    x_train, x_test, y_train, y_test, test_df = prep.train_test_split(x,y)
    model = LstmBrandDetector()
    print('Model fitting...')
    model.create_model()
    model.fit(x_train, y_train, epochs=8)
    print('Accuracy for whole titles: {}'.format(model.evaluate(x_test, y_test)))
    preds = model.predict(x_test, test_df)
    preds.to_csv('helper_files/predictions.csv')