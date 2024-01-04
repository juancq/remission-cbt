import numpy as np
SEED = 1
np.random.seed(SEED)
import random
random.seed(SEED)

import itertools
import yaml
from types import SimpleNamespace

import pandas as pd
import shap
import matplotlib.pylab as pl

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

from sklearn import preprocessing
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import pickle

from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.models import NodeConfig
from pytorch_tabular.models import TabNetModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig

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


data_config = DataConfig(
	target=['target'], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
	continuous_cols=features,
	#categorical_cols=cat_col_names,
	)

trainer_config = TrainerConfig(
	#auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
	batch_size=64,
	max_epochs=50,
	gpus=None, #index of the GPU to use. -1 means all available GPUs, None, means CPU
	)


optimizer_config = OptimizerConfig()

y_variable='pranxfree'
params_set = pickle.load(open(f'pytorch_params_10fold_{y_variable}.pk', 'rb'))
params_set = params_set['params']


#k_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=123)

test_set = []
shap_set = []
k_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
for i, (train,test) in enumerate(k_folds.split(df.to_numpy(), df[y_variable].to_numpy())):
    #if i > 5:
    #    break
    #if i <= 5:
    #    continue
        
    scaler = StandardScaler()
    x = scaler.fit_transform(df.iloc[train][features].to_numpy())
    df_x = pd.DataFrame(x, columns=features)
    df_x[y_variable] = df[y_variable].iloc[train].to_numpy()

    df_train, df_val = train_test_split(df_x, test_size=0.10, stratify=df_x[y_variable])
    x_test = scaler.transform(df.iloc[test][features].to_numpy())
    df_test = pd.DataFrame(x_test, columns=features)
    df_test[y_variable] = df.iloc[test][y_variable].to_numpy()
    params = params_set[i]
    
    data_config.target = [y_variable]
    trainer_config.batch_size = params['batch_size']

    model_config = NodeConfig(
            task="classification",
            num_trees = 2**(params['num_trees'] + 4),
            num_layers = params['num_layers'],
            learning_rate = params['lr'],
            )
    model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            )

    model.fit(train=df_train, validation=df_val)

    def f_anxcat(__X):
        _d = pd.DataFrame(__X, columns=df_train.columns)
        pred_df = model.predict(_d)
        return pred_df['1.0_probability'].to_numpy().flatten()


    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    node_diag_explainer = shap.KernelExplainer(f_anxcat, shap.sample(df_train, 400))
    print(df_test.shape)
    shap_values = node_diag_explainer.shap_values(df_test, nsamples=200)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    test_set.append(test)
    shap_set.append(shap_values)

test_set = np.concatenate(test_set)
shap_set = np.concatenate(shap_set)

pickle.dump({'test':test_set, 'shap': shap_set}, open(f'shap_{y_variable}.pk', 'wb'))
