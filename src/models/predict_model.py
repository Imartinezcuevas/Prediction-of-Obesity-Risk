import numpy as np
import pandas as pd
import pickle
import logging

def predict(model, encoder, test):
    y_pred_test = model.predict(test)
    result = np.ravel(encoder.inverse_transform(y_pred_test.reshape(-1, 1)))

    results = pd.DataFrame({
    'NObeyesdad': result
    })

    final = pd.concat([test, results], axis=1)
    final.to_csv('./data/results.csv', index=False)


def main():
    """ Predicts the data (saved in ../processed/y.csv) with the model (saved in ../models/stacking_model.pkl).
    """
    file = open('./models/encoder.pkl', 'rb')
    encoder = pickle.load(file)
    file.close()

    file = open('./models/stacking_model.pkl', 'rb')
    model = pickle.load(file)
    file.close()
     
    test =  pd.read_csv("./data/processed/test.csv")

    predict(model, encoder, test)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()