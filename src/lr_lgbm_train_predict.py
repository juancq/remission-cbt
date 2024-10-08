import warnings
warnings.filterwarnings("ignore")
import argparse
import pickle
import itertools
import neptune
import yaml
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

import numpy as np
SEED = 1
np.random.seed(SEED)
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from sklearn import metrics
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

from netcal.scaling import TemperatureScaling
import netcal.metrics as netcal_metrics


import xgboost as xgb
import optuna

from lightgbm import LGBMClassifier

from ECE import ece_score


neptune_object = neptune.init('--set to path')

def get_models_and_distributions():
    models = [
        ('lr', lambda: LogisticRegression(solver='saga', max_iter=500)),
        ('lgbm', lambda: LGBMClassifier()),
    ]
    
    distributions = {
        'lr': {
            'C': optuna.distributions.LogUniformDistribution(1e-3, 100),
            'penalty': optuna.distributions.CategoricalDistribution(['l2', 'l1', 'elasticnet', 'none']),
            'l1_ratio': optuna.distributions.CategoricalDistribution([0.1, 0.2, 0.5, 0.7, 0.9]),
        },
        'lgbm': {
            'max_depth': optuna.distributions.IntUniformDistribution(3, 30),
            'feature_fraction': optuna.distributions.UniformDistribution(0.1, 1.0),
            'learning_rate': optuna.distributions.LogUniformDistribution(0.005, 0.6),
            'num_leaves': optuna.distributions.IntUniformDistribution(10, 300),
            "subsample": optuna.distributions.CategoricalDistribution([0.6, 0.8, 1.0]),
            "reg_alpha": optuna.distributions.UniformDistribution(0.01, 9),
            "reg_lambda": optuna.distributions.UniformDistribution(0.01, 9),
        }
    }
    
    return models, distributions


def update_scores(scores, y_val, preds, preds_prob):
    scores['roc'].append(metrics.roc_auc_score(y_val, preds_prob))
    scores['f1'].append(metrics.f1_score(y_val, preds))
    scores['acc'].append(metrics.accuracy_score(y_val, preds))
    precision, recall, _ = metrics.precision_recall_curve(y_val, preds_prob)
    scores['pr_auc'].append(metrics.auc(recall, precision))
    scores['ece'].append(ece_score(preds_prob, y_val))


def main():
    '''
    Main function to execute the machine learning pipeline.

    This function orchestrates the entire machine learning workflow:
    1. Parses command-line arguments and loads configuration.
    2. Initializes experiment tracking with Neptune (if enabled).
    3. Loads and preprocesses the dataset.
    4. Iterates over specified outcome variables:
        a. Prepares data for the current outcome.
        b. Trains and evaluates multiple machine learning models:
            - Performs hyperparameter optimization.
            - Conducts cross-validation.
            - Calibrates models.
            - Computes various performance metrics.
    5. Logs results and best parameters for each model and outcome.
    6. Tracks total execution time.

    The function supports multiple model types (e.g., Logistic Regression, LightGBM)
    and can handle various preprocessing steps like imputation and encoding.
    Results are logged both to console and to Neptune (if enabled) for each
    model-outcome combination.
    '''
    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='runtime configurations', type=str)
    parser.add_argument("-v", help="verbose", action='store_true')
    parser.add_argument("--kfold", action='store_true')
    parser.add_argument("--tag", help="tags for neptune", type=str)
    parser.add_argument("--temperature", action='store_true')
    parser.add_argument("--no_neptune", action='store_true')


    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        #config = SimpleNamespace(**config)

    df_train = pd.read_csv(config['data_missing_cases'], low_memory=False)
    #df_train = df_train.sample(frac=0.70, random_state=123)

    if config['model']:
        features = list(itertools.chain.from_iterable(config['features'][i] for i in config['model']))
    else:
        features = list(itertools.chain.from_iterable(config['features'].values()))

    categorical = config['categorical']

    outcomes = config['outcomes']

    df_train = df_train[features+config['outcomes']]

    if config['impute']:
        imp = SimpleImputer(strategy='most_frequent')
        df_train[categorical] = imp.fit_transform(df_train[categorical])
        imp_it = IterativeImputer(random_state=0, max_iter=20, estimator=BayesianRidge())
        numeric_features = list(set(features) - set(categorical))
        df_train[numeric_features] = imp_it.fit_transform(df_train[numeric_features])
        df_train[numeric_features] = df_train[numeric_features].round(0).astype(int)
    else:
        df_train = df_train.dropna(subset=features)

    enc = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(df_train[categorical])

    categorical_names = enc.get_feature_names().tolist()
    df_train[categorical_names] = enc.transform(df_train[categorical])
    df_train = df_train.drop(columns=categorical)
    features = df_train.drop(columns=config['outcomes']).columns.tolist()

    models, distributions = get_models_and_distributions()

    net_ece = netcal_metrics.ECE(10)

    for y_variable in config['outcomes']:
        X = df_train.drop(columns=config['outcomes']).values
        Y = df_train[y_variable].astype(int).values

        print(y_variable)
        print(len(Y))
        print(Y.sum())
        print(Y.sum()/len(Y))

        print('^'*30)
        print(f'y_variable outcome {y_variable}')
        for model_name, model_cls in models:
            scores = defaultdict(list)

            if not args.no_neptune:
                neptune_params = {'model': model_name, 'outcome':y_variable, 
                    'features':','.join(config['features'].keys())}
                neptune.create_experiment(name='train-test-val', params=neptune_params)
                for tag in config['tags']:
                    neptune.append_tag(tag)

            print(model_name)
            print('roc, f1, acc')
            y_truth = []
            y_preds = []
            params_set = []
            if args.kfold:
                k_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
                if not args.no_neptune:
                    neptune.append_tag('10folds')
            else:
                k_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=123)
            for i, (train,test) in enumerate(k_folds.split(X, Y)):
                scaler = StandardScaler()
                X[train] = scaler.fit_transform(X[train])
                x_train = X[train]
                y_train = Y[train]
                x_train, x_val, y_train, y_val = train_test_split(X[train], Y[train], 
                                                    test_size=0.10, stratify=Y[train])
                # obtain set for calibration
                #x_val, x_cal, y_val, y_cal = train_test_split(x_val, y_val,
                #                                    test_size=0.4, stratify=y_val)

                x_test = scaler.transform(X[test])
                y_test = Y[test]

                # hyperparameter search
                search = optuna.integration.OptunaSearchCV(model_cls(),
                            param_distributions=distributions[model_name],
                            n_trials=config['n_trials'],
                            timeout=config['timeout'],
                            scoring='roc_auc',
                            #n_jobs=-1
                            )
                search.fit(x_train, y_train)
                #model = search

                params_set.append(search.best_params_)

                # calibration
                if args.temperature:
                    model = search
                    confidences = model.predict_proba(x_val)[:,1]
                    temperature = TemperatureScaling()
                    temperature.fit(confidences, y_val)
                else:
                    model = CalibratedClassifierCV(search, method="isotonic", cv="prefit")
                    model.fit(x_val, y_val)

                x_val = x_test
                y_val = y_test
                # model evaluation
                preds_prob_vec = model.predict_proba(x_val)
                preds_prob = preds_prob_vec[:,1]

                if args.temperature:
                    preds_prob = temperature.transform(preds_prob)
                    preds = (preds_prob > 0.5).astype(int)
                else:
                    preds = model.predict(x_val)

                update_scores(scores, y_val, preds, preds_prob)
            
                y_truth.append(y_val)
                y_preds.append(preds)
            y_truth = np.concatenate(y_truth)
            y_preds = np.concatenate(y_preds)
            #y_truth = Y_val
            #y_preds = preds

            print(metrics.confusion_matrix(y_truth, y_preds))

            for key,val in scores.items():
                _mean = np.mean(val)
                _std = np.std(val)
                print(f'{key}  {_mean:.2f} ({_std:.2f})')
                if not args.no_neptune:
                    neptune.log_metric(key, _mean)
                    neptune.log_metric(key+'_std', _std)
                #for _c in val:
                #    neptune.log_metric(key+'_combined', _c)

            #if args.tag:
            #    neptune.append_tag(args.tag)
            #neptune.append_tag([model_name, 'validation'])
            if not args.no_neptune:
                neptune.stop()

            pickle.dump(params_set, 
                    open(f'params_{y_variable}_{model_name}.pk', 'wb'))

    print(f"Total execution time: {datetime.now() - start_time}")

if __name__ == "__main__":
    main()
