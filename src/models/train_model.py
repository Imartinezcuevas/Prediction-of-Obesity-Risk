import click
import logging
import warnings
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import pickle

X_train = pd.read_csv("./data/processed/splits/X_train.csv")
X_test = pd.read_csv("./data/processed/splits/X_test.csv")
y_train = np.asarray(pd.read_csv("./data/processed/splits/y_train.csv")).ravel()
y_test = np.asarray(pd.read_csv("./data/processed/splits/y_test.csv")).ravel()

def RF(trial):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        criterion=trial.suggest_categorical("criterion", ['gini', 'entropy', 'log_loss']),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 100),
        max_depth=trial.suggest_int("max_depth", 1, 100),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 100),
        random_state=27
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def XGB(trial):
    model = XGBClassifier(
        max_depth=trial.suggest_int('max_depth', 1, 100),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        n_estimators=trial.suggest_int('n_estimators', 50, 1000),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 10),
        gamma=trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        subsample=trial.suggest_float('subsample', 0.01, 1.0, log=True),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.01, 1.0, log=True),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=27
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def Cat(trial):
    model = CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 100, 1000),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 100),
        depth=trial.suggest_int("depth", 7, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        verbose=False,
        random_state=27
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def LGBM(trial):
    model = LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        max_depth=trial.suggest_int("max_depth", 1, 100),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        objective='multiclass',
        verbosity=-1,
        boosting_type=trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        num_leaves=trial.suggest_int('num_leaves', 2, 256),
        min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
        random_state=27
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

@click.command()
def main():
    """ Train base models with tuned hyperparameters
    """
    logger = logging.getLogger(__name__)
    logger.info('Training modelos with tuned hyperparameters')

    base_models = []

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=27)

    
    study_rf = optuna.create_study(study_name="random_forest", direction="maximize", sampler=sampler)
    study_rf.optimize(RF, n_trials=100)

    print("Number of finished trials: ", len(study_rf.trials))
    print("Best trial:")
    trial = study_rf.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print()
    base_models.append(("random_forest", RandomForestClassifier().set_params(**study_rf.best_params)))

    ## XGBoost
    study_xgb = optuna.create_study(study_name="xgb", direction="maximize", sampler=sampler)
    study_xgb.optimize(XGB, n_trials=100)

    print("Number of finished trials: ", len(study_xgb.trials))
    print("Best trial:")
    trial = study_xgb.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print()
    base_models.append(("xgb", XGBClassifier().set_params(**study_xgb.best_params)))

    ## CatBoost
    study_catboost = optuna.create_study(study_name="catboost", direction="maximize", sampler=sampler)
    study_catboost.optimize(Cat, n_trials=100)

    print("Number of finished trials: ", len(study_catboost.trials))
    print("Best trial:")
    trial = study_catboost.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print()
    base_models.append(("catboost", CatBoostClassifier().set_params(**study_catboost.best_params)))

    ## LGBM
    study_lgbm = optuna.create_study(study_name="lgbm", direction="maximize", sampler=sampler)
    study_lgbm.optimize(LGBM, n_trials=20)

    print("Number of finished trials: ", len(study_lgbm.trials))
    print("Best trial:")
    trial = study_lgbm.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print()
    base_models.append(("lgbm", LGBMClassifier().set_params(**study_lgbm.best_params)))

    meta_model = XGBClassifier().set_params(**study_xgb)

    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
    stacking_model.fit(X_train, y_train)

    y_pred_test = stacking_model.predict(X_test)
    accuracy_val = accuracy_score(y_test, y_pred_test)
    print(f"Validation Accuracy Score: {accuracy_val:.8f}")

    pickle.dump(stacking_model, open("./models/stacking_model.pkl", "wb"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()