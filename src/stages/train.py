import argparse
import joblib
import pandas as pd
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from typing import Dict, Text


def train(df: pd.DataFrame, target_column: Text, param_grid: Dict, cv: int, n_iter: int, seed: int):
    """Train model.
    Args:
        df {pandas.DataFrame}: dataset
        target_column {Text}: target column name
        param_grid {Dict}: grid parameters
        cv {int}: cross-validation value
        n_iter {int}: number of iteration for the RandomizedSearchCV
        seed {int}: Random seed
    Returns:
        trained model
    """

    estimator = LogisticRegression()
    f1_scorer = make_scorer(f1_score, average='weighted')

    # Perform randomized search cross-validation
    clf = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        random_state=seed,
        verbose=1,
        scoring=f1_scorer
    )
    # Get X and Y
    y_train = df.loc[:, target_column].values.astype('int32')
    X_train = df.drop(target_column, axis=1).values.astype('float32')
    clf.fit(X_train, y_train)

    return clf


def train_model(config_path: Text) -> None:
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    train_df = pd.read_csv(config['data_split']['trainset_path'])

    model = train(
        df=train_df,
        target_column=config['featurize']['target_column'],
        param_grid=config['train']['param_grid'],
        cv=config['train']['cv'],
        n_iter=config['train']['n_iter'],
        seed=config['base']['random_state']
    )
    models_path = config['train']['model_path']
    joblib.dump(model, models_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
