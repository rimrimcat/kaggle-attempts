try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    pass

from enum import Enum, auto
from math import ceil
from pprint import pprint
from typing import Callable, List, Optional, Tuple, Type, Union

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna import Trial
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.split import ExpandingSlidingWindowSplitter
from tqdm import tqdm

from __scripts__.cross_val import CombinatorialPurgedKFold
from __scripts__.data import Task


def base_est_regressor():
    return [
        ("rf1", RandomForestRegressor()),
        ("rf2", RandomForestRegressor()),
        ("rf3", RandomForestRegressor()),
    ]


def base_est_classifier():
    return [
        ("rf1", RandomForestClassifier()),
        ("rf2", RandomForestClassifier()),
        ("rf3", RandomForestClassifier()),
    ]


BASE_REGRESSORS = [
    KNeighborsRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor(),
    ExtraTreesRegressor(),
    BaggingRegressor(),
    VotingRegressor(base_est_regressor()),
    StackingRegressor(base_est_regressor()),
    MLPRegressor(max_iter=2000),
    GaussianProcessRegressor(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    # SVR(),
]
BASE_CLASSIFIERS = [
    KNeighborsClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(),
    BaggingClassifier(),
    VotingClassifier(base_est_classifier()),
    StackingClassifier(base_est_classifier()),
    MLPClassifier(max_iter=2000),
    GaussianNB(),
    BernoulliNB(),
    GaussianProcessClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(reg_param=0.5),
    # SVC(),
]


class ModelClassElement:
    _value_: int
    model_cls: Type
    default_params: dict
    task: Task
    supports_partial_fit: bool


class ModelClass(ModelClassElement, Enum):
    # trunk-ignore-all(mypy/var-annotated)
    RF_CLASSIFIER = auto(), RandomForestClassifier, {}, Task.clf_sl(), False
    MLP_CLASSIFIER = auto(), MLPClassifier, {"max_iter": 2000}, Task.clf_ml(), True
    LDA = auto(), LinearDiscriminantAnalysis, {}, Task.clf_sl(), False

    def __new__(
        cls,
        value: int,
        model_cls: Type,
        default_params: dict,
        task: Task,
        supports_partial_fit: bool,
    ):
        obj = ModelClassElement.__new__(cls)
        obj._value_ = value
        obj.model_cls = model_cls
        obj.task = task
        obj.supports_partial_fit = supports_partial_fit
        return obj

    @classmethod
    def _missing_(cls, value):
        try:
            return cls[value]
        except KeyError:
            pass

        try:
            code = int(value)  # In case value is a string representation of a number
            for item in cls:
                if item._value_ == code:
                    return item
                if item.model_cls == value:
                    return item
        except (ValueError, TypeError):
            pass

        raise ValueError(f"'{value}' is not a valid {cls.__name__}")


class ModelParams:
    @staticmethod
    def decision_tree(
        trial: Trial,
        task: Task,
    ):
        match task.task:
            case "regression":
                criterion = trial.suggest_categorical(
                    "criterion",
                    ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                )
            case "classification":
                criterion = trial.suggest_categorical(
                    "criterion",
                    ["gini", "entropy", "log_loss"],
                )
            case _:
                raise ValueError(f"Unknown task: {task}")

        return {
            "criterion": criterion,
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 2, 32, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 1000),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 1000),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0.0, 0.5
            ),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2"]
            ),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 32, log=True),
            "min_impurity_decrease": trial.suggest_float(
                "min_impurity_decrease", 0.0, 1.0
            ),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 1.0),
        }

    @staticmethod
    def lda(
        trial: Trial,
        task: Task,
        n_features: int,
        n_classes: int,
    ):
        paramd = {
            "solver": "lsqr",
            "n_components": trial.suggest_int(
                "n_components", 1, min(n_classes - 1, n_features)
            ),
        }

        shrinkage_or_oa = trial.suggest_categorical(
            "shrinkage_or_oa", [None, "auto", "oa"]
        )

        match shrinkage_or_oa:
            case "auto":
                paramd.update({"shrinkage": "auto"})
            case "oa":
                paramd.update(
                    {
                        "covariance_estimator": OAS(
                            store_precision=False, assume_centered=False
                        )
                    }
                )

            case None:
                paramd.update({"shrinkage": None})

        return paramd

    @staticmethod
    def mlp(
        trial: Trial,
        task: Task,
        num_hidden_layers: Optional[int] = None,
        max_iter: int = 2000,
    ):
        if not num_hidden_layers:
            num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 10)

        hidden_layer_sizes: list[int] = []
        for i in range(num_hidden_layers):
            hidden_layer_sizes.append(trial.suggest_int(f"layer {i} size", 1, 100))

        return {
            "hidden_layer_sizes": tuple(hidden_layer_sizes),
            "activation": trial.suggest_categorical(
                "activation", ["identity", "logistic", "tanh", "relu"]
            ),
            # "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
            "max_iter": max_iter,
        }

    @staticmethod
    def random_forest(trial: Trial, task: Task):
        match task.task:
            case "regression":
                criterion = trial.suggest_categorical(
                    "criterion",
                    ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                )
            case "classification":
                criterion = trial.suggest_categorical(
                    "criterion",
                    ["gini", "entropy", "log_loss"],
                )
            case _:
                raise ValueError(f"Unknown task: {task}")

        return {
            "criterion": criterion,
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "max_depth": trial.suggest_categorical(
                "max_depth", [2, 4, 8, 16, 32, None]
            ),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2"]
            ),
            "random_state": 69,
        }

    @staticmethod
    def random_forest_all(trial: Trial, task: Task):
        match task.task:
            case "regression":
                criterion = trial.suggest_categorical(
                    "criterion",
                    ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                )
            case "classification":
                criterion = trial.suggest_categorical(
                    "criterion",
                    ["gini", "entropy", "log_loss"],
                )
            case _:
                raise ValueError(f"Unknown task: {task}")

        no_max_depth = trial.suggest_categorical("no_max_depth", [True, False])
        if no_max_depth:
            max_depth = None
        else:
            max_depth = trial.suggest_int("max_depth", 2, 32)

        return {
            "criterion": criterion,
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "max_depth": max_depth,
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2"]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            ),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-5, 1e-1, log=True),
            "oob_score": trial.suggest_categorical("oob_score", [True, False]),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0.0, 0.5
            ),
            "min_impurity_decrease": trial.suggest_float(
                "min_impurity_decrease", 0.0, 0.5
            ),
            "random_state": 69,
        }

    @staticmethod
    def adaboost(
        trial: Trial,
        task: Task,
    ):
        estimator_params = ModelParams.decision_tree(trial, task)

        match task.task:
            case "regression":
                base_estimator = DecisionTreeRegressor(**estimator_params)
                pass
            case "classification":
                base_estimator = DecisionTreeClassifier(**estimator_params)
                pass
            case _:
                raise ValueError(f"Unknown task: {task}")
                pass

        return {
            "estimator": base_estimator,
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 1.0),
            "random_state": 69,
        }


def min_of_mean_median(scores: list[float]):
    return np.min([np.mean(scores), np.median(scores)])


def kfold_cv_iter(
    model,
    X,
    Y,
    scoring,
    overall_scoring=min_of_mean_median,
    n_splits: int = 5,
    shuffle: bool = True,
):
    kf = KFold(n_splits=n_splits, shuffle=shuffle)

    scores = []
    for step, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Train model
        model.fit(X_train, Y_train)

        # Get score for this fold
        fold_score = scoring(model, X_test, Y_test)
        scores.append(fold_score)

        yield (step, overall_scoring(scores))


def kfold_cv(
    model,
    X,
    Y,
    scoring,
    overall_scoring=min_of_mean_median,
    n_splits: int = 5,
    shuffle: bool = True,
    iterate: bool = False,
):
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    cv_scores = cross_val_score(model, X, Y, scoring=scoring, cv=kf)
    return overall_scoring(cv_scores)


def comb_purged_kfold_cv_iter(
    model,
    X,
    Y,
    scoring,
    overall_scoring=min_of_mean_median,
    n_splits: int = 5,
    shuffle: bool = True,
):
    cpcv = CombinatorialPurgedKFold(
        n_groups=int(X.shape[0] / 150),
        test_groups=2,
        frac_embargo=0.01,
    )
    scores = []
    for step, (train_idx, test_idx) in enumerate(cpcv.split(X)):
        # Split data for this fold

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Train model
        model.fit(X_train, Y_train)

        # Get score for this fold
        fold_score = scoring(model, X_test, Y_test)
        scores.append(fold_score)

        yield (step, overall_scoring(scores))


def comb_purged_kfold_cv(
    model,
    X,
    Y,
    scoring,
    overall_scoring=min_of_mean_median,
    n_splits: int = 5,
    shuffle: bool = True,
):
    cpcv = CombinatorialPurgedKFold(
        n_groups=int(X.shape[0] / 150),
        test_groups=2,
        frac_embargo=0.01,
    )
    splits = cpcv.split(X)
    print("Got splits:", splits)
    cv_scores = cross_val_score(model, X, Y, scoring=scoring, cv=splits)
    return overall_scoring(cv_scores)


def opt_base_model(X, Y, task: Task, do_cv: bool = True):
    match task.task:
        case "regression":
            model_list = BASE_REGRESSORS
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
        case "classification":
            model_list = BASE_CLASSIFIERS
            # scorer = make_scorer(roc_auc_score)
            scorer = make_scorer(accuracy_score)

    if not task.multi_label:
        Y = Y.ravel()

    overall_scores = []

    if task.time_series and do_cv:

        def train_test_model(model):
            overall_scores.append(comb_purged_kfold_cv(model, X, Y, scorer))

    elif do_cv:

        def train_test_model(model):
            overall_scores.append(kfold_cv(model, X, Y, scorer))

    else:

        def train_test_model(model):
            model.fit(X, Y)
            overall_scores.append(model.score(X, Y))

    tq = tqdm(model_list, desc="Evaluating models...")
    for i, model in enumerate(tq):
        if i == 0:
            prev_model = ""
            tq.set_description(f"{type(model).__qualname__}: evaluating...")
        else:
            tq.set_description(
                f"{prev_model}: {overall_scores[i-1]:.4f} |  {type(model).__qualname__}: evaluating.."
            )

        train_test_model(model)

        prev_model = type(model).__qualname__

    # for model in model_list:
    #     print("Evaluating model:", type(model).__qualname__)

    #     if task.time_series and do_cv:
    #         overall_scores.append(comb_purged_kfold_cv(model, X, Y, scorer))
    #     elif do_cv:
    #         overall_scores.append(kfold_cv(model, X, Y, scorer))
    #     else:
    #         model.fit(X, Y)
    #         overall_scores.append(model.score(X, Y))

    trial_df = pd.DataFrame(
        {
            "model": [type(model).__qualname__ for model in model_list],
            "score": overall_scores,
            "model_index": range(len(model_list)),
        }
    )

    trial_df.sort_values(by="score", ascending=False, inplace=True)
    trial_df.reset_index(drop=True, inplace=True)
    print(trial_df[["model", "score"]])

    best_mdl = model_list[trial_df.loc[0, "model_index"]]
    best_mdl.fit(X, Y)

    return best_mdl


def tune_model(
    X,
    Y,
    task: Task,
    model_cls: Type[Union[RegressorMixin, ClassifierMixin]],
    param_func: Callable[..., dict],
    study_name: Optional[str] = None,
    total_trials: int = 1000,
    n_jobs: int = 8,
    storage: str = "sqlite:///example.db",
):
    if not task.multi_label:
        Y = Y.ravel()

    match task.task:
        case "classification":
            scorer = make_scorer(roc_auc_score)
        case _:
            raise NotImplementedError()

    # mcls = ModelClass(model_cls)

    def model_search_objective(trial: optuna.Trial):
        model = model_cls(**param_func(trial))

        for step, score in kfold_cv_iter(model, X, Y, scorer):
            trial.report(score, step)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return score

    def optimize_study(study_name: str, n_trials):
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=TPESampler(
                multivariate=True,
                warn_independent_sampling=False,
            ),
            pruner=HyperbandPruner(),
            load_if_exists=True,
        )
        study.optimize(model_search_objective, n_trials=n_trials)

    study = optuna.create_study(
        storage=storage,
        direction="maximize",
        study_name=study_name,
        load_if_exists=True,
    )

    joblib.Parallel(n_jobs)(
        joblib.delayed(optimize_study)(
            study.study_name, n_trials=ceil(total_trials / n_jobs)
        )
        for _ in range(n_jobs)
    )

    best_params = param_func(study.best_trial)
    best_model = model_cls(**best_params)
    best_model.fit(X, Y)

    print("BEST PARAMETERS:")
    pprint(best_params)

    return best_model
