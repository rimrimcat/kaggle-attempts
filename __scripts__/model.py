try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    pass

from IPython import get_ipython
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import (
    ElasticNet,
    GammaRegressor,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    PoissonRegressor,
    Ridge,
    SGDClassifier,
    SGDRegressor,
    TweedieRegressor,
)

if get_ipython() is None:
    from tqdm import tqdm
else:
    # This fixes nested loops if in IPython
    from tqdm.notebook import tqdm

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
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.split import ExpandingSlidingWindowSplitter
import logging
from __scripts__.cross_val import CombinatorialPurgedKFold
from __scripts__.data import Task

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s %(message)s",
    handlers=[logging.StreamHandler()],
)

# mute the matplotlib logger
matplotlib_logger = logging.getLogger("matplotlib.category")
matplotlib_logger.setLevel(logging.WARN)
lightgbm_logger = logging.getLogger("lightgbm")
lightgbm_logger.setLevel(logging.WARN)

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
    model_name: str


class ModelClass(ModelClassElement, Enum):

    # trunk-ignore-all(mypy/var-annotated)
    # Classification Models
    RF_CLF = auto(), RandomForestClassifier, {}, Task.clf_sl(), False
    MLP_CLF = auto(), MLPClassifier, {"max_iter": 2000}, Task.clf_ml(), True
    LDA = auto(), LinearDiscriminantAnalysis, {}, Task.clf_sl(), False
    QDA = (
        auto(),
        QuadraticDiscriminantAnalysis,
        {"reg_param": 0.5},
        Task.clf_ml(),
        False,
    )
    KNN_CLF = auto(), KNeighborsClassifier, {}, Task.clf_ml(), False
    ADA_BOOST_CLF = auto(), AdaBoostClassifier, {}, Task.clf_sl(), False
    GAUSSIAN_NB = auto(), GaussianNB, {}, Task.clf_sl(), False
    BERNOULLI_NB = auto(), BernoulliNB, {}, Task.clf_sl(), False
    EXTRA_TREES_CLF = auto(), ExtraTreesClassifier, {}, Task.clf_sl(), False
    SCALABLE_SVC = (
        auto(),
        BaggingClassifier,
        {"estimator": SVC(max_iter=2000)},
        Task.clf_sl(),
        False,
        "ScalableSVC",
    )
    GRADIENT_BOOSTING_CLF = (
        auto(),
        GradientBoostingClassifier,
        {},
        Task.clf_sl(),
        False,
    )
    LOGISTIC_REGRESSION = (
        auto(),
        LogisticRegression,
        {"max_iter": 2000, "solver": "lbfgs"},
        Task.clf_sl(),
        True,
    )
    DECISION_TREE_CLF = auto(), DecisionTreeClassifier, {}, Task.clf_sl(), False
    SVC = auto(), SVC, {"kernel": "rbf", "max_iter": 4000}, Task.clf_sl(), False
    LINEAR_SVC = (
        auto(),
        LinearSVC,
        {"max_iter": 2000, "dual": "auto"},
        Task.clf_sl(),
        False,
    )
    # MULTINOMIAL_NB doesnt want negative values
    # MULTINOMIAL_NB = auto(), MultinomialNB, {}, Task.clf_sl(), False
    # COMPLEMENT_NB, IDK
    # COMPLEMENT_NB = auto(), ComplementNB, {}, Task.clf_sl(), False
    SGD_CLF = (
        auto(),
        SGDClassifier,
        {"max_iter": 1000, "tol": 1e-3},
        Task.clf_sl(),
        True,
    )
    PASSIVE_AGGRESSIVE_CLF = (
        auto(),
        PassiveAggressiveClassifier,
        {"max_iter": 1000},
        Task.clf_sl(),
        True,
    )
    # XGBOOST_CLF = (
    #     auto(),
    #     XGBClassifier,
    #     {"n_estimators": 100, "learning_rate": 0.1},
    #     Task.clf_sl(),
    #     False,
    #     "XGBoostClassifier",
    # )
    LIGHTGBM_CLF = (
        auto(),
        LGBMClassifier,
        {"n_estimators": 100},
        Task.clf_sl(),
        False,
        "LightGBMClassifier",
    )
    VOTING_CLF = (
        auto(),
        VotingClassifier,
        {"estimators": base_est_classifier()},
        Task.clf_sl(),
        False,
    )
    STACKING_CLF = (
        auto(),
        StackingClassifier,
        {"estimators": base_est_classifier()},
        Task.clf_sl(),
        False,
    )

    # Regression Models
    RF_REGR = auto(), RandomForestRegressor, {}, Task.regr_sl(), False
    MLP_REGR = auto(), MLPRegressor, {"max_iter": 2000}, Task.regr_ml(), True
    KNN_REGR = auto(), KNeighborsRegressor, {}, Task.regr_ml(), False
    ADA_BOOST_REGR = auto(), AdaBoostRegressor, {}, Task.regr_sl(), False
    GRADIENT_BOOSTING_REGR = (
        auto(),
        GradientBoostingRegressor,
        {},
        Task.regr_sl(),
        False,
    )
    EXTRA_TREES_REGR = auto(), ExtraTreesRegressor, {}, Task.regr_sl(), False
    BAGGING_REGR = auto(), BaggingRegressor, {}, Task.regr_sl(), False
    VOTING_REGR = (
        auto(),
        VotingRegressor,
        {"estimators": base_est_regressor()},
        Task.regr_sl(),
        False,
    )
    STACKING_REGR = (
        auto(),
        StackingRegressor,
        {"estimators": base_est_regressor()},
        Task.regr_sl(),
        False,
    )
    LINEAR_REGRESSION = auto(), LinearRegression, {}, Task.regr_sl(), False
    RIDGE = auto(), Ridge, {"alpha": 1.0}, Task.regr_sl(), True
    LASSO = auto(), Lasso, {"alpha": 1.0, "max_iter": 1000}, Task.regr_sl(), True
    ELASTIC_NET = (
        auto(),
        ElasticNet,
        {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 1000},
        Task.regr_sl(),
        True,
    )
    SVR = auto(), SVR, {"kernel": "rbf", "max_iter": 2000}, Task.regr_sl(), False
    LINEAR_SVR = (
        auto(),
        LinearSVR,
        {"max_iter": 2000, "dual": "auto"},
        Task.regr_sl(),
        False,
    )
    DECISION_TREE_REGR = auto(), DecisionTreeRegressor, {}, Task.regr_sl(), False
    SGD_REGR = (
        auto(),
        SGDRegressor,
        {"max_iter": 1000, "tol": 1e-3},
        Task.regr_sl(),
        True,
    )
    PASSIVE_AGGRESSIVE_REGR = (
        auto(),
        PassiveAggressiveRegressor,
        {"max_iter": 1000},
        Task.regr_sl(),
        True,
    )
    HUBER_REGR = (
        auto(),
        HuberRegressor,
        {"max_iter": 100, "epsilon": 1.35},
        Task.regr_sl(),
        False,
    )
    POISSON_REGR = auto(), PoissonRegressor, {"max_iter": 100}, Task.regr_sl(), False
    GAMMA_REGR = auto(), GammaRegressor, {"max_iter": 100}, Task.regr_sl(), False
    TWEEDIE_REGR = (
        auto(),
        TweedieRegressor,
        {"power": 1.5, "max_iter": 100},
        Task.regr_sl(),
        False,
    )
    # XGBOOST_REGR = (
    #     auto(),
    #     XGBoostReg,
    #     {"n_estimators": 100, "learning_rate": 0.1},
    #     Task.regr_sl(),
    #     False,
    # )
    LIGHTGBM_REGR = (
        auto(),
        LGBMRegressor,
        {"n_estimators": 100},
        Task.regr_sl(),
        False,
        "LightGBMRegressor",
    )

    def __new__(
        cls,
        value: int,
        model_cls: Type,
        default_params: dict,
        task: Task,
        supports_partial_fit: bool,
        model_name: Optional[str] = None,
    ):
        obj = ModelClassElement.__new__(cls)
        obj._value_ = value
        obj.model_cls = model_cls
        obj.default_params = default_params
        obj.task = task
        obj.supports_partial_fit = supports_partial_fit
        if not model_name:
            model_name = model_cls.__name__
        obj.model_name = model_name
        return obj

    def instantiate(self):
        return self.model_cls(**self.default_params)

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
                    print("_missing_ compares value==code:", item._value_ == code)
                    return item
                if item.model_cls == value:
                    print(
                        "_missing_ compares model_cls==value:", item.model_cls == value
                    )
                    return item
        except (ValueError, TypeError):
            pass

        raise ValueError(f"'{value}' is not a valid {cls.__name__}")

    @classmethod
    def get_models_for_task(cls, task: Task) -> list["ModelClass"]:
        model_list = []
        for mdl in cls:
            if mdl.task.task == task.task and (
                not task.multi_label or task.multi_label and mdl.task.multi_label
            ):
                model_list.append(mdl)
        return model_list


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
    cpcv: Optional[CombinatorialPurgedKFold] = None,
    n_groups: Optional[int] = None,
    test_groups: int = 2,
    frac_embargo: float = 0.01,
):
    if not cpcv:
        cpcv = CombinatorialPurgedKFold(
            n_groups=int(X.shape[0] / 150) if not n_groups else n_groups,
            test_groups=test_groups,
            frac_embargo=frac_embargo,
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
):
    cpcv = CombinatorialPurgedKFold(
        n_groups=int(X.shape[0] / 200),
        test_groups=2,
        frac_embargo=0.01,
    )

    cv_scores = cross_val_score(model, X, Y, scoring=scoring, cv=cpcv)
    return overall_scoring(cv_scores)


def opt_base_model(X, Y, task: Task, do_cv: bool = True):
    modelclass_list = ModelClass.get_models_for_task(task)
    scorer = task.get_scorer()

    if not task.multi_label:
        Y = Y.ravel()

    overall_scores = []

    if task.time_series and do_cv:

        def train_test_model(model):
            cpcv = CombinatorialPurgedKFold(
                n_groups=5,
                test_groups=2,
                frac_embargo=0.01,
            )
            cv_scores = []

            tq2 = tqdm(
                comb_purged_kfold_cv_iter(
                    model,
                    X,
                    Y,
                    scorer,
                    cpcv=cpcv,
                ),
                desc="Starting...",
                total=cpcv.n_splits,
                leave=False,
            )

            for step, score in tq2:
                cv_scores.append(score)
                tq2.set_description(
                    f"CV split {step+1}/{cpcv.n_splits}: {min_of_mean_median(cv_scores):.4f}"
                )

            overall_scores.append(min_of_mean_median(cv_scores))

    elif do_cv:

        def train_test_model(model):
            cv = KFold(
                n_splits=5,
                shuffle=True,
            )
            cv_scores = []

            tq2 = tqdm(
                kfold_cv_iter(
                    model,
                    X,
                    Y,
                    scorer,
                    cv=cv,
                ),
                desc="Starting...",
                total=cv.n_splits,
                leave=False,
            )

            for step, score in tq2:
                cv_scores.append(score)
                tq2.set_description(
                    f"CV split {step + 1}/{cv.n_splits}: {min_of_mean_median(cv_scores):.4f}"
                )

            overall_scores.append(min_of_mean_median(cv_scores))

    else:

        def train_test_model(model):
            model.fit(X, Y)
            overall_scores.append(model.score(X, Y))

    tq = tqdm(modelclass_list, desc="Evaluating models...")
    for i, model_cls_item in enumerate(tq):

        # Instantiate model first!
        model = model_cls_item.instantiate()

        if i == 0:
            prev_model = ""
            tq.set_description(f"{type(model).__qualname__}: evaluating...")
        else:
            tq.set_description(
                f"{prev_model}: {overall_scores[i-1]:.4f} |  {type(model).__qualname__}: evaluating.."
            )

        train_test_model(model)

        if i == len(modelclass_list) - 1:
            tq.set_description("Done!")
        else:
            prev_model = type(model).__qualname__

    trial_df = pd.DataFrame(
        {
            "model": [
                # type(modelcls_item.instantiate()).__qualname__
                modelcls_item.model_name
                for modelcls_item in modelclass_list
            ],
            "score": overall_scores,
            "model_index": range(len(modelclass_list)),
        }
    )

    trial_df.sort_values(by="score", ascending=False, inplace=True)
    trial_df.reset_index(drop=True, inplace=True)
    print(trial_df[["model", "score"]])

    best_mdl = modelclass_list[trial_df.loc[0, "model_index"]].instantiate()
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
