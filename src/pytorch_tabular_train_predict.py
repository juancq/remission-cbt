import warnings
warnings.filterwarnings("ignore")
import argparse
import itertools
import pickle
import neptune
import yaml
from collections import defaultdict
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
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
import optuna

from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.models import NodeConfig
from pytorch_tabular.models import TabNetModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.utils import get_balanced_sampler, get_class_weighted_cross_entropy

from ECE import ece_score
from netcal.scaling import TemperatureScaling
import netcal.metrics as netcal_metrics

neptune_object = neptune.init('--set to path')

PROB_LABEL_POS='1.0_probability'
PROB_LABEL_NEG='0.0_probability'

def remission_trial(trial, df, y_variable, features):
    data_config = DataConfig(
            target=[y_variable], 
            continuous_cols=features,
            #categorical_cols=cat_col_names,
            )

    trainer_config = TrainerConfig(
            batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
            max_epochs=20,
            gpus=None, #index of the GPU to use. -1 means all available GPUs, None, means CPU
            )

    optimizer_config = OptimizerConfig()

    #num_trees = trial.suggest_categorical('num_trees', [32, 64])#, 128, 256, 512, 1024])
    num_trees = 2**(trial.suggest_int('num_trees', 1, 5) + 4)
    if num_trees < 256:
        num_layers = trial.suggest_int('num_layers', 1, 2)
    else:
        num_layers = trial.suggest_int('num_layers', 1, 1)
    
    model_config = NodeConfig(
            task="classification",
            num_trees = num_trees,
            num_layers = num_layers,
            learning_rate = trial.suggest_categorical("lr", [0.1, 0.01, 0.001, 0.0001]),
            )

    model_cls = lambda: TabularModel(
            data_config=data_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
            model_config=model_config,
            )

    score = []
    k_folds = StratifiedKFold(n_splits=3, shuffle=True)
    for train,test in k_folds.split(df.to_numpy(), df[y_variable].to_numpy()):
        scaler = StandardScaler()
        x = scaler.fit_transform(df.iloc[train][features].to_numpy())
        df_x = pd.DataFrame(x, columns=features)
        df_x[y_variable] = df[y_variable].iloc[train].to_numpy()

        df_train, df_val = train_test_split(df_x, test_size=0.10, stratify=df_x[y_variable])
        #print(df_train[y_variable].sum())
        #print(df_val[y_variable].sum())

        model = model_cls()
        model.fit(train=df_train, validation=df_val)

        # model evaluation
        pred_df = model.predict(df_val)
        y_test = df_val[y_variable]
        _roc = metrics.roc_auc_score(y_test, pred_df[PROB_LABEL_POS])
        score.append(_roc)

    return np.mean(score)



def main():
    '''
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='runtime configurations', type=str)
    parser.add_argument("-v", help="verbose", action='store_true')
    parser.add_argument("--kfold", action='store_true')
    parser.add_argument("--tag", help="tags for neptune", type=str)
    parser.add_argument("--temperature", action='store_true')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        #config = SimpleNamespace(**config)

    if args.tag:
        config['tags'].append(args.tag)

    df = pd.read_csv(config['data_missing_cases'], low_memory=False)

    if config['model']:
        features = list(itertools.chain.from_iterable(config['features'][i] for i in config['model']))
    else:
        features = list(itertools.chain.from_iterable(config['features'].values()))

    categorical = config['categorical']

    df = df[features+config['outcomes']]

    cat = categorical
    if config['impute']:
        imp = SimpleImputer(strategy='most_frequent')
        df[cat] = imp.fit_transform(df[cat])
        imp_it = IterativeImputer(random_state=0, max_iter=20, estimator=BayesianRidge())
        numeric_features = list(set(features) - set(cat))
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

    model_name = 'PytorchTabular'


    for y_variable in config['outcomes']:
        print(y_variable)

        data_config.target = [y_variable]
        print('^'*30)
        print(f'y_variable outcome {y_variable}')

        scores = defaultdict(list)

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
        if args.temperature:
            neptune.append_tag('temp')
        if args.kfold:
            k_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
            neptune.append_tag('10folds')
        else:
            k_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=123)
        print(df[config['outcomes']].isnull().any())
        for i, (train,test) in enumerate(k_folds.split(df.to_numpy(), df[y_variable].to_numpy())):
            scaler = StandardScaler()
            x = scaler.fit_transform(df.iloc[train][features].to_numpy())
            df_x = pd.DataFrame(x, columns=features)
            df_x[y_variable] = df[y_variable].iloc[train].to_numpy()

            df_train, df_val = train_test_split(df_x, test_size=0.10, stratify=df_x[y_variable])
            #df_val, df_test = train_test_split(df_val, test_size=0.70, stratify=df_val[y_variable])

            x_test = scaler.transform(df.iloc[test][features].to_numpy())
            df_test = pd.DataFrame(x_test, columns=features)
            df_test[y_variable] = df.iloc[test][y_variable].to_numpy()

            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: remission_trial(trial, df_train, y_variable, features), n_trials=20)#, timeout=500)

            params = study.best_trial.params
            params_set.append(params)
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

            if args.temperature:
                pred_df = model.predict(df_val)
                proba = pred_df[PROB_LABEL_POS].to_numpy()
                temperature = TemperatureScaling()
                temperature.fit(proba, df_val[y_variable].to_numpy())

            # model evaluation
            pred_df = model.predict(df_test)
            proba = pred_df[PROB_LABEL_POS]
            y_test = df_test[y_variable]

            if args.temperature:
                proba = temperature.transform(proba.to_numpy()).flatten()
                preds = (proba > 0.5).astype(int)
                proba_vec = np.zeros((len(preds), 2))
                proba_vec[:,1] = proba
                proba_vec[:,0] = 1-proba
            else:
                proba_vec = pred_df[[PROB_LABEL_NEG, PROB_LABEL_POS]].values
                preds = pred_df.prediction.to_numpy()

            _roc = metrics.roc_auc_score(y_test, proba)
            _f1 = metrics.f1_score(y_test, preds)
            _acc = metrics.accuracy_score(y_test, preds)

            precision, recall, thresholds = metrics.precision_recall_curve(y_test, proba)
            r = metrics.auc(recall, precision)

            scores['roc'].append(_roc)
            scores['f1'].append(_f1)
            scores['acc'].append(_acc)
            scores['pr_auc'].append(r)
            #_ece = ece_score(pred_df[[PROB_LABEL_NEG, PROB_LABEL_POS]].values, y_test)
            _ece = ece_score(proba_vec, y_test)
            scores['ece'].append(_ece)

            #neptune.log_metric('roc', _roc)
            #neptune.log_metric('f1', _f1)
            #neptune.log_metric('acc', _acc)
            #neptune.log_metric('pr_auc', r)
            #neptune.log_metric('ece', ece_score(pred_df[['0_probability', '1_probability']].values, y_test))
        
            y_truth.append(y_test)
            y_preds.append(pred_df.prediction)
        y_truth = np.concatenate(y_truth)
        y_preds = np.concatenate(y_preds)

        print(metrics.confusion_matrix(y_truth, y_preds))

        for key,val in scores.items():
            _mean = np.mean(val)
            _std = np.std(val)
            print(f'{key}  {_mean:.2f} ({_std:.2f})')
            neptune.log_metric(key, _mean)
            neptune.log_metric(key+'_std', _std)
        neptune.stop()

        pickle.dump({'params':params_set, 'folds':k_folds}, 
            open(f'pytorch_params_10fold_{y_variable}_longer.pk', 'wb'))


if __name__ == "__main__":
    main()
