import pandas as pd
import click
import logging
from sklearn.model_selection import train_test_split
import os

def get_variable_types(dataframe):
    continuous_vars = []
    categorical_vars = []

    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            categorical_vars.append(column)
        else:
            continuous_vars.append(column)

    return continuous_vars, categorical_vars

def calculate_bmr(weight, height, age, is_male):
  """
  Calculates the BMR based on the revised Harris-Benedict equation.

  Args:
    weight: Weight in kilograms.
    height: Height in centimeters.
    age: Age in years.
    is_male: 0 if female, 1 if male.

  Returns:
    The BMR value.
  """
  if is_male:
    return (447.6 + 9.25 * weight) + (3.10 * height * 100) - 4.33 * age
  else:
    return (88.4 + 13.4 * weight) + (4.8 * height * 100) - 5.68 * age


@click.command()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    print(os.getcwd())
    train = pd.read_csv("./data/raw/train.csv")
    test = pd.read_csv("./data/raw/test.csv")

    continuous_vars, categorical_vars = get_variable_types(train)
    continuous_vars.remove('id'), categorical_vars.remove('NObeyesdad')

    train = pd.get_dummies(train, columns=categorical_vars, drop_first=True)
    test = pd.get_dummies(test, columns=categorical_vars, drop_first=True)

    train['BMI'] = train['Weight'] / train['Height']**2
    test['BMI'] = test['Weight'] / test['Height']**2
    train['BMR'] = train.apply(lambda row: calculate_bmr(row['Weight'], row['Height'], row['Age'], row['Gender_Male']), axis=1)
    test['BMR'] = test.apply(lambda row: calculate_bmr(row['Weight'], row['Height'], row['Age'], row['Gender_Male']), axis=1)

    X = train.drop(['NObeyesdad'], axis=1)
    y = train['NObeyesdad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    train.to_csv('./data/processed/train.csv', index=False)
    test.to_csv('./data/processed/test.csv', index=False)
    y.to_csv('./data/processed/y.csv', index=False)  

    os.makedirs('./data/processed/splits')
    X_train.to_csv('./data/processed/splits/X_train.csv', index=False)
    X_test.to_csv('./data/processed/splits/X_test.csv', index=False)
    y_train.to_csv('./data/processed/splits/y_train.csv', index=False)
    y_test.to_csv('./data/processed/splits/y_test.csv', index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
