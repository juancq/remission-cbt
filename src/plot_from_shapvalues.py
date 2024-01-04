import numpy as np
SEED = 1
np.random.seed(SEED)
import random
random.seed(SEED)

import itertools
import pickle
import shap
import yaml
import pandas as pd
import matplotlib.pylab as plt

from types import SimpleNamespace

from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def main():

    with open('classify_remission_with_predictors_optuna_final.yml', 'r') as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config['data_missing_cases'], low_memory=False)

    if config['model']:
        features = list(itertools.chain.from_iterable(config['features'][i] for i in config['model']))
    else:
        features = list(itertools.chain.from_iterable(config['features'].values()))
    categorical = config['categorical']
    df = df[features+config['outcomes']]


    if config['impute']:
        imp = SimpleImputer(strategy='most_frequent')
        df[categorical] = imp.fit_transform(df[categorical])
        imp_it = IterativeImputer(random_state=0, max_iter=20, estimator=BayesianRidge())
        numeric_features = list(set(features) - set(categorical))
        df[numeric_features] = imp_it.fit_transform(df[numeric_features])
        df[numeric_features] = df[numeric_features].round(0).astype(int)
    else:
        df = df.dropna(subset=features)

    enc = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(df[categorical])

    categorical_names = enc.get_feature_names().tolist()
    df[categorical_names] = enc.transform(df[categorical])
    df = df.drop(columns=categorical)
    features = df.drop(columns=config['outcomes']).columns.tolist()

    y_variable='pranxfree'
    #y_variable='anxcat'

    shap_set = pickle.load(open(f'shap_{y_variable}.pk', 'rb'))
    shap_values = shap_set['shap']
    test_values = shap_set['test']

    head = 0
    k_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    test_set = []
    for i, (train,test) in enumerate(k_folds.split(df.to_numpy(), df[y_variable].to_numpy())):
        scaler = StandardScaler()
        x = scaler.fit_transform(df.iloc[train][features].to_numpy())
        df_x = pd.DataFrame(x, columns=features)
        df_x[y_variable] = df[y_variable].iloc[train].to_numpy()

        x_test = scaler.transform(df.iloc[test][features].to_numpy())
        df_test = pd.DataFrame(x_test, columns=features)
        df_test[y_variable] = df.iloc[test][y_variable].to_numpy()

        #print(test)
        #print(test_values[head:head+len(test)])
        #head += len(test)
        test_set.append(df_test)

    test_set = pd.concat(test_set)
    print(test_set.shape)

    replace_cat_names = {f'x{i}': _category for i,_category in enumerate(config['categorical'])}
    readable_names = test_set.columns.to_series()
    for k, v in replace_cat_names.items():
        readable_names = readable_names.str.replace(k, v)
        readable_names.tolist()

    shap.summary_plot(shap_values=shap_values, features=test_set,
                        feature_names=readable_names, show=False)

    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.savefig(f'final_shap_plot_{y_variable}.png', bbox_inches='tight', dpi=400)




if __name__ == '__main__':
    main()
