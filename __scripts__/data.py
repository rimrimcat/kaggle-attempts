import logging
from collections import UserDict
from dataclasses import dataclass
from enum import Enum, auto
from math import ceil
from types import MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    Union,
    cast,
)

from missforest import MissForest

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import float64
from numpy.typing import NDArray
from pandas.core.frame import DataFrame
from scipy.stats import skew
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute._base import _BaseImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.preprocessing._encoders import _BaseEncoder

from __scripts__.plot import (
    plot_bivariate_violin,
    plot_counts,
    plot_monovariate_violin,
    plot_pca_loadings,
    plot_principal_components,
    plot_scree,
)
from __scripts__.types import BasicDataType, DataType, DataTypeDict, Task

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s %(message)s",
    handlers=[logging.StreamHandler()],
)

# mute the matplotlib logger
matplotlib_logger = logging.getLogger("matplotlib.category")
matplotlib_logger.setLevel(logging.WARN)

if TYPE_CHECKING:
    from numpy._core.multiarray import _Array
    from scipy.sparse import csr_array, csr_matrix

    _X_t: TypeAlias = (
        _Array[tuple[Any | int, Literal[0]], float64]
        | Any
        | csr_matrix
        | csr_array
        | NDArray
    )
else:
    _X_t: TypeAlias = Any

transform_item: TypeAlias = Union[BaseEstimator, _BaseEncoder, str]
transform_list: TypeAlias = list[
    tuple[str, transform_item, Union[list[str], slice, list[int]]]
]

dft_fcn_col: TypeAlias = Callable[[Any], Any]
dft_fcn_cols: TypeAlias = Callable[[pd.Series], Any]
dft_fcn_colmany: TypeAlias = Callable[[pd.DataFrame], Any]
df_transform_list: TypeAlias = Sequence[
    Union[
        tuple[dft_fcn_col, str],
        tuple[dft_fcn_col, str, str],
        tuple[Union[dft_fcn_cols, dft_fcn_colmany], list[str], str],
        tuple[Union[dft_fcn_cols, dft_fcn_colmany], list[str], list[str]],
    ]
]

outer_f_in_f: TypeAlias = Union[
    Callable[[pd.DataFrame], Union[dft_fcn_cols, dft_fcn_colmany]],
    Callable[[pd.Series], dft_fcn_col],
]


def onehot_to_ordinal(X):
    return np.where(
        (X == 0).all(axis=1)[:, None],
        np.nan,
        np.argmax(X, axis=1)[:, None],
    )


def onehot_to_ordinal_new(X: NDArray):
    n_rows, _ = X.shape

    result = np.full(n_rows, np.nan)

    for i in range(n_rows):
        ones_indices = np.where(X[i] == 1)[0]

        if len(ones_indices) == 1:
            result[i] = ones_indices[0]

    return result.reshape(-1, 1)


def make_matrix_to_df(col_names: list[str]):
    def matrix_to_df(X):
        return pd.DataFrame({col: X[:, i] for i, col in enumerate(col_names)})

    return matrix_to_df


def df_to_matrix(X: pd.DataFrame):
    return X.to_numpy()


def make_ordinal_to_onehot(num_categories: int):
    def ordinal_to_onehot(X):
        n_samples, n_features = X.shape

        onehot = np.zeros((n_samples, num_categories), dtype=int)
        not_nan = ~np.isnan(X)

        row_indices = np.where(not_nan)[0]
        col_values = X[not_nan].astype(int)

        for row, col in zip(row_indices, col_values):
            if 0 <= col < num_categories:
                onehot[row, col] = 1
        return onehot

    return ordinal_to_onehot


class FunctionInFunction:
    f_in_f: outer_f_in_f
    inner_f: Union[dft_fcn_col, dft_fcn_cols]

    def __init__(self, f_in_f: outer_f_in_f) -> None:
        self.f_in_f = f_in_f

    def fit(self, X: Any):
        self.inner_f = self.f_in_f(X)

    def __call__(self, X: Union[pd.Series, Any]) -> Any:
        return self.inner_f(X)


class DataFrameTransformer:
    transformers: df_transform_list

    def __init__(
        self,
        transformers: df_transform_list,
    ) -> None:
        self.transformers = transformers

    def fit(self, df: pd.DataFrame):
        for tup in self.transformers:
            trans = tup[0]

            if hasattr(trans, "fit"):
                trans.fit(df[tup[1]])

    def transform(self, df: pd.DataFrame):
        for tup in self.transformers:
            in_cols = tup[1]

            if len(tup) == 3:
                out_cols = tup[2]
            else:
                out_cols = in_cols

            trans = tup[0]

            if isinstance(in_cols, list) and len(in_cols) > 1:
                output = trans(df[in_cols])
            else:
                output = df[in_cols].apply(trans)
            df[out_cols] = output
        return df

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)


class MissForestTransformer:
    mf: MissForest

    def __init__(self, mf: MissForest) -> None:
        self.mf = mf

    def fit(self, X, y=None, **params):
        return self.mf.fit(X)

    def transform(self, X, **params):
        return self.mf.transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None, **params) -> pd.DataFrame:
        return self.mf.fit_transform(X)


class ColumnTransformer(_ColumnTransformer):
    """
    ColumnTransform that handles inverse transforms

    Adapted from: https://github.com/scikit-learn/scikit-learn/issues/11463#issuecomment-1674435238
    """

    def inverse_transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        arrays = []
        for name, trans, _column, _weight in self._iter(
            fitted=True,
            column_as_labels=False,
            skip_drop=False,
            skip_empty_columns=False,
        ):
            ind_dict = self.output_indices_  # {name: indices}
            if trans not in (None, "passthrough", "drop"):
                arr = trans.inverse_transform(
                    X[:, ind_dict[name].start : ind_dict[name].stop]
                )

            elif trans in ("drop",):
                arr = np.zeros(
                    (X.shape[0], ind_dict[name].stop - ind_dict[name].start + 1)
                )

            elif trans in ("passthrough",):
                arr = X[:, ind_dict[name].start : ind_dict[name].stop]

            arrays.append(arr)

        retarr = np.concatenate(arrays, axis=1)
        return retarr

    def inverse_transform_df(self, X):
        X_t = self.inverse_transform(X)
        col_names = [
            name for name in self.output_indices_.keys() if name != "remainder"
        ]

        pd_dict = {name: X_t[:, i] for i, name in enumerate(col_names)}

        return pd.DataFrame(pd_dict)

    def fit_transform(
        self,
        X,
        y=None,
        missforest=None,
        simple_imputer=None,
        transform_then_impute=True,
        **fit_params,
    ) -> _X_t:
        X_t = super().fit_transform(X, y, **fit_params)

        if missforest:
            if isinstance(missforest, MissForest):
                pass
            elif missforest is True and hasattr(self, "_dtype_dict"):
                dtype_dict = self._dtype_dict
                categorical_ = [
                    col for col, d in dtype_dict.items() if d.is_categorical()
                ]

                drop_slice = slice(0, 0, None)

                trans_names = [x[0] for x in self._transformers]
                categorical = [
                    col
                    for col in categorical_
                    if col in self.output_indices_
                    and self.output_indices_[col] != drop_slice
                ]

                missforest = MissForest(
                    categorical=categorical,
                    max_iter=25,
                    verbose=0,
                )

            mf = MissForestTransformer(missforest)
            self._imputer = ColumnTransformer._create_column_missforest_imputer(
                mf, self.output_indices_, X_t
            )

            X_t = self._imputer.fit_transform(X_t)

        elif simple_imputer:
            # Create trans_dict
            if isinstance(simple_imputer, _BaseImputer) or isinstance(
                simple_imputer, BaseEstimator
            ):
                trans_dict = {
                    k: clone(simple_imputer) for k in self.output_indices_.keys()
                }
            elif isinstance(simple_imputer, dict):
                trans_dict = simple_imputer
            else:
                raise ValueError(f"Unknown type for imputer: {type(simple_imputer)}")

            self._imputer = ColumnTransformer._create_column_simple_imputer(
                trans_dict,
                self.output_indices_,
                X_t,
            )

            X_t = self._imputer.fit_transform(X_t)

        return X_t

    def transform(self, X, **params):
        X_t = super().transform(X, **params)
        if hasattr(self, "_imputer"):
            X_t = self._imputer.transform(X_t)
        return X_t

    @staticmethod
    def _create_column_transformer(
        df: pd.DataFrame,
        dtype_dict: dict[str, DataType],
        trans_dict: Optional[dict[str, Union[BaseEstimator, _BaseEncoder]]] = None,
    ):
        transformers: transform_list = []

        for col in df.columns:
            if trans_dict and col in trans_dict:
                transformers.append(
                    (
                        col,
                        trans_dict[col],
                        [col],
                    )
                )
                continue

            dtype: DataType = dtype_dict[col]

            if dtype.is_continuous():
                transformers.append(
                    (
                        col,
                        StandardScaler(),
                        [col],
                    )
                )
            elif dtype.is_binary() or dtype.is_ordinal():
                categories = df[col].dropna().unique()
                categories.sort()
                transformers.append(
                    (
                        col,
                        OrdinalEncoder(
                            categories=[categories],
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                            encoded_missing_value=np.nan,
                        ),
                        [col],
                    )
                )
            elif not dtype.is_ordinal():
                categories = df[col].dropna().unique()
                transformers.append(
                    (
                        col,
                        OneHotEncoder(
                            categories=[categories],
                            handle_unknown="ignore",
                        ),
                        [col],
                    )
                )

        return ColumnTransformer(transformers=transformers)

    @staticmethod
    def _create_column_missforest_imputer(
        mf: MissForestTransformer,
        output_ind_dict: dict[str, slice],
        X_t: _X_t,
    ):
        fw_transformers: transform_list = []
        rev_transformers: transform_list = []

        for i, (col, _slice) in enumerate(output_ind_dict.items()):
            if col == "remainder":
                continue

            slice_len = _slice.stop - _slice.start

            if not slice_len:
                # skip dropped column
                continue

            if slice_len > 1:
                X_slice = X_t[:, _slice]
                num_categories = X_slice.shape[1]

                fw_transformers.append(
                    (
                        col,
                        FunctionTransformer(onehot_to_ordinal),
                        _slice,
                    )
                )
                rev_transformers.append(
                    (
                        col,
                        FunctionTransformer(make_ordinal_to_onehot(num_categories)),
                        [i],
                    )
                )
            else:
                fw_transformers.append(
                    (
                        col,
                        "passthrough",
                        _slice,
                    )
                )
                rev_transformers.append((col, "passthrough", [i]))

        drop_slice = slice(0, 0, None)

        col_names = [
            key
            for key, _slice in output_ind_dict.items()
            if key != "remainder" and _slice != drop_slice
        ]

        fw_trans = _ColumnTransformer(transformers=fw_transformers)
        m2df = FunctionTransformer(make_matrix_to_df(col_names))
        df2m = FunctionTransformer(df_to_matrix)
        rev_trans = _ColumnTransformer(transformers=rev_transformers)

        pipe = Pipeline(
            [
                ("fw", fw_trans),
                ("m2df", m2df),
                ("imputer", mf),
                ("df2m", df2m),
                ("rev", rev_trans),
            ]
        )

        return pipe

    @staticmethod
    def _create_column_simple_imputer(
        trans_dict: dict[str, Union[BaseEstimator, _BaseEncoder]],
        output_ind_dict: dict[str, slice],
        X_t: _X_t,
    ):
        transformers: transform_list = []

        for col, _slice in output_ind_dict.items():
            if col == "remainder":
                continue

            fcn: transform_item = "passthrough"
            if col in trans_dict:
                fcn = trans_dict[col]
            elif "remainder" in trans_dict:
                fcn = trans_dict["remainder"]
            else:
                logger = logging.getLogger(
                    "model.ColumnTransform._create_column_imputer"
                )
                logger.info(
                    "NOTE: 'remainder' not specified. Defaulting to 'passthrough'."
                )

            # Check if multiple slice
            slice_len = _slice.stop - _slice.start
            if slice_len > 1:
                # OneHotEncoded value
                X_slice = X_t[:, _slice]
                num_categories = X_slice.shape[1]

                fw_trans = FunctionTransformer(onehot_to_ordinal)
                rev_trans = FunctionTransformer(make_ordinal_to_onehot(num_categories))

                pipe = Pipeline(
                    [
                        ("fw", fw_trans),
                        ("imputer", fcn),
                        ("rev", rev_trans),
                    ]
                )

                transformers.append(
                    (
                        col,
                        pipe,
                        _slice,
                    )
                )
            else:
                transformers.append(
                    (
                        col,
                        fcn,
                        _slice,
                    )
                )

        return _ColumnTransformer(
            transformers=transformers
        )  # does not have inverse transform

    @staticmethod
    def create_transformers(
        df: pd.DataFrame,
        dtype_dict: dict[str, DataType],
        labels: Union[str, list[str]],
        trans_dict: Optional[dict[str, Union[BaseEstimator, _BaseEncoder]]] = None,
    ) -> tuple["ColumnTransformer", "ColumnTransformer", Task]:
        if isinstance(labels, str):
            labels = [labels]

        task = Task(
            multi_label=True if len(labels) > 1 else False,
            task=(
                "classification"
                if dtype_dict[labels[0]].is_categorical()
                else "regression"
            ),
        )

        x_cols = [col for col in df.columns if col not in labels]
        df_X = df[x_cols]
        df_Y = df[labels]

        trans_X = ColumnTransformer._create_column_transformer(
            df_X, dtype_dict, trans_dict
        )
        trans_Y = ColumnTransformer._create_column_transformer(
            df_Y, dtype_dict, trans_dict
        )

        trans_X._dtype_dict = dtype_dict
        trans_Y._dtype_dict = dtype_dict

        return (trans_X, trans_Y, task)


def summarize_data(df: pd.DataFrame, print_summary: bool = True):
    """Returns summary statistics about the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        print (bool): Whether to print the summary statistics.

    Returns:
        dict: A dictionary containing the summary statistics.

    Prints:
        The size of the DataFrame, the number of columns and rows containing NaNs,
        the number of total NaNs, and the data type for each column.
    """

    cols = df.columns
    n_cols = len(cols)
    n_rows = len(df)

    nan_cols = {
        col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()
    }
    nan_rows = df.isna().any(axis=1).sum()
    nans = df.isna().sum().sum()

    # Create subdf for data types
    data_types = []
    values = []
    for col in cols:
        dtype = DataType.infer_ser_dtype(df[col])
        data_types.append(dtype)

        unique_values = df[col].dropna().unique()
        unique_values.sort()
        uv = unique_values.tolist()
        value = ""
        match dtype:
            case (
                DataType.CATEGORICAL_NOMINAL_BINARY
                | DataType.CATEGORICAL_ORDINAL_BINARY
            ):
                value = f"{uv[0]}, {uv[1]}"
            case (
                DataType.CATEGORICAL_NOMINAL_STRING
                | DataType.CATEGORICAL_ORDINAL_STRING
            ):
                value = ", ".join(uv)
            case DataType.CATEGORICAL_INTEGER:
                value = ", ".join([str(v) for v in uv])
            case DataType.NUMERIC:
                n_unique = df[col].dropna().unique()
                if isinstance(n_unique.min(), float):
                    value = f"min: {round(n_unique.min(), 2)}, max: {round(n_unique.max(), 2)}, x̄: {df[col].mean().round(2)}, σ: {df[col].std().round(2)}"
                else:
                    value = f"min: {n_unique.min().round(2)}, max: {n_unique.max().round(2)}, x̄: {df[col].mean().round(2)}, σ: {df[col].std().round(2)}"
            case _:
                value = ", ".join([str(u) for u in uv])

        if len(value) > 50 and not dtype.is_continuous():
            value = f"({len(uv)} unique values) {value}"
            if value.endswith("..."):
                pass
            elif value.endswith(".."):
                value += "."
            elif value.endswith("."):
                value += ".."
            else:
                value = value[:47] + "..."

        values.append(value)

    data_types_df = pd.DataFrame(
        {
            "Column": cols,
            "Data Type": [d.name for d in data_types],
            "Values": values,
            "Notes": ["" for d in data_types],
        }
    )

    summary_dict = {
        "n_cols": n_cols,
        "n_rows": n_rows,
        "nan_cols": nan_cols,
        "nan_rows": nan_rows,
        "nans": nans,
        "data_types": data_types,
        "data_types_df": data_types_df,
    }

    if print_summary:
        print("=#=#=#=#=#=#=#=#= SUMMARY =#=#=#=#=#=#=#=#=")

        print("=== SIZE ===")
        print(f"{n_cols} columns")
        print(f"{n_rows} rows")
        print("")

        print("=== INCOMPLETENESS ===")
        print(f"{len(nan_cols)} columns containing NaNs: {nan_cols}")
        print(f"{nan_rows} rows containing NaNs")
        print(f"{nans} total NaNs")
        print("")

        print("=== DATA TYPES ===")
        print(data_types_df)
        print("")

    return summary_dict


def clean_data(
    df: DataFrame,
    print_summary: bool = True,
    drop_uninformative: bool = True,
    drop_missing_rows: bool = False,
    add_new_features: bool = False,
) -> DataFrame:
    """Cleans the dataframe by removing columns with unknown data types.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        print_summary (bool): Whether to print the summary of the cleaned DataFrame.
        drop_uninformative (bool): Whether to drop columns with uninformative data.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only columns with a non-unknown data type.

    Prints:
        The size of the DataFrame, the number of columns and rows containing NaNs,
        the number of total NaNs, and the data type for each column.
    """
    logger = logging.getLogger("data.clean_data")

    summary_0 = summarize_data(df, print_summary=False)

    if add_new_features:
        added_features = []
        added_transforms = []
        for d, col in zip(summary_0["data_types"], df.columns):
            if d.is_continuous():
                sk_val = skew(df[col], nan_policy="omit")
                if sk_val > 2:

                    def log_tform(X: pd.Series):
                        out = np.log(X)
                        out[out == -np.inf] = np.nan
                        return out

                    new_feature = f"Log{col}"
                    out = np.log(df[col])
                    out[out == -np.inf] = np.nan

                    df[new_feature] = out

                    added_transforms.append((log_tform, [col], new_feature))
                    added_features.append(new_feature)

    summary_1 = summarize_data(df, print_summary=False)

    if drop_uninformative:
        df = df[[col for dt, col in zip(summary_1["data_types"], df.columns) if dt]]

    if drop_missing_rows:
        df = df.dropna()

    summary_2 = summarize_data(df, print_summary=False)

    if print_summary:
        if drop_uninformative:
            mod_data_types_df: pd.DataFrame = summary_1["data_types_df"]
            mod_data_types_df["Notes"] = [
                "DROPPED" if d.should_drop else "" for d in summary_1["data_types"]
            ]
            if add_new_features and added_features:
                for ft in added_features:
                    mod_data_types_df.loc[
                        mod_data_types_df["Column"] == ft, "Notes"
                    ] = "ADDED"

            i2col = {
                str(k): v
                for k, v in summary_1["data_types_df"].to_dict()["Column"].items()
            }
            col2newi = {
                v: str(k)
                for k, v in summary_2["data_types_df"].to_dict()["Column"].items()
            }

            new_indices = [
                "" if d.should_drop else col2newi[i2col[ind]]
                for ind, d in zip(
                    mod_data_types_df.index.astype(str), summary_1["data_types"]
                )
            ]
            mod_data_types_df.index = new_indices

        print("=#=#=#=#=#=#=#=#= SUMMARY =#=#=#=#=#=#=#=#=")

        print("=== SIZE ===")
        print(f"{summary_0['n_cols']} -> {summary_2['n_cols']} columns")
        print(f"{summary_0['n_rows']} -> {summary_2['n_rows']} rows")
        print("")

        print("=== INCOMPLETENESS ===")
        nan_col_str = f": {summary_2['nan_cols']}" if summary_2["nan_cols"] else ""
        print(
            f"{len(summary_0['nan_cols'])} -> {len(summary_2['nan_cols'])} columns containing NaNs{nan_col_str}"
        )
        print(
            f"{summary_0['nan_rows']} -> {summary_2['nan_rows']} rows containing NaNs"
        )
        print(f"{summary_0['nans']} -> {summary_2['nans']} total NaNs")
        print("")

        print("=== DATA TYPES ===")
        print(summary_1["data_types_df"].to_string())
        print("")

    if summary_2["nans"]:
        logger.info(
            "NOTE: NaN values may be present in data. Either drop missing values or specify an imputer for the Column Transformer."
        )

    if add_new_features:
        logger.info(
            "NOTE: New features were added. Remember to add the transforms to existing dataframe"
        )

        return df, added_transforms
    else:
        return df


def check_corr(
    df: pd.DataFrame,
    label: str,
    dtype_dict: dict[str, DataType],
    num_plots_x: int = 3,
):
    x_cols = df.columns.drop(label).to_list()
    y = label

    n_plots = len(x_cols)

    plot_counts(df, dtype_dict, num_plots_x, ceil((n_plots + 1) / num_plots_x))

    plot_bivariate_violin(
        df, dtype_dict, x_cols, y, num_plots_x, ceil(n_plots / num_plots_x)
    )


def do_pca(
    x_data: NDArray,
    y_data: Optional[NDArray] = None,
    x_trans: Optional[_ColumnTransformer] = None,
    y_trans: Optional[_ColumnTransformer] = None,
    task: Optional[Task] = None,
    class_names: Optional[list[str]] = None,
    n_components: Optional[int] = None,
    component_display_limit: int = 8,
    plot_type: Literal["2d", "3d"] = "2d",
):
    pca = PCA(n_components=n_components)

    all_index = None

    match x_data, y_data, x_trans, y_trans:
        case np.ndarray(), np.ndarray(), _ColumnTransformer(), _ColumnTransformer():
            all_data = np.concatenate((x_data, y_data), axis=1)
            all_index = [x.split("__")[-1] for x in x_trans.get_feature_names_out()] + [
                x.split("__")[-1] for x in y_trans.get_feature_names_out()
            ]

        case np.ndarray(), None, _ColumnTransformer(), None:
            all_data = x_data
            all_index = [x.split("__")[-1] for x in x_trans.get_feature_names_out()]

        case np.ndarray(), np.ndarray(), None, None:
            all_data = np.concatenate((x_data, y_data), axis=1)

        case np.ndarray(), None, None, None:
            all_data = x_data

        case _, _, _, _:
            raise ValueError("Invalid inputs!")

    pca.fit(all_data)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
        index=all_index,
    )

    plot_scree(pca, component_display_limit=component_display_limit)

    # print(loadings.loc["Survived", :].sort_values(ascending=False))
    # print(loadings.loc[:, "PC11"].sort_values(ascending=False))

    projected_data = pca.transform(all_data)

    if task:
        if task.task == "classification" and y_data is not None:
            if class_names:
                pass
            elif y_trans and len(y_trans._transformers) == 1:
                class_names = []

                for name, tform in y_trans._transformers:
                    match tform:
                        case OrdinalEncoder():
                            tform = cast(OrdinalEncoder, tform)
                            class_names += [
                                f"{name}={str(x)}" for x in tform.categories[0].tolist()
                            ]

                            pass
                        case _:
                            pass

            plot_principal_components(
                projected_data[:, 0],
                projected_data[:, 1],
                classes=y_data,
                class_names=class_names,
            )

            plot_pca_loadings(
                pca,
                all_index,
            )

    # 2D PCA Plot
    if plot_type == "2d":
        # fig = plt.figure(figsize=(10, 6))
        # ax = fig.add_subplot(111)
        # for i in range(pca.components_.shape[0]):
        #     ax.arrow(
        #         0,
        #         0,
        #         pca.components_[i, 0],
        #         pca.components_[i, 1],
        #         color="r",
        #         alpha=0.5,
        #         head_width=0.05,
        #         head_length=0.1,
        #     )
        #     ax.text(
        #         pca.components_[i, 0] * 1.15,
        #         pca.components_[i, 1] * 1.15,
        #         loadings.index[i],
        #         color="g",
        #         ha="center",
        #         va="center",
        #     )
        #
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_xlabel("PC1")
        # ax.set_ylabel("PC2")
        # ax.set_title("PCA Loadings Plot")
        # ax.grid(True)
        # ax.axhline(0, color="black", linewidth=0.8)
        # ax.axvline(0, color="black", linewidth=0.8)
        # plt.show(block=False)
        pass

    elif plot_type == "3d":
        raise NotImplementedError()
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection="3d")
        # ax.scatter(
        #     projected_data[:, 0], projected_data[:, 1], projected_data[:, 2], alpha=0.7
        # )
        # ax.set_xlabel("PC1")
        # ax.set_ylabel("PC2")
        # ax.set_zlabel("PC3")
        # ax.set_title("3D PCA Projection")
        # plt.show(block=False)
        #
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection="3d")
        # for i in range(pca.components_.shape[0]):
        #     ax.quiver(
        #         0,
        #         0,
        #         0,
        #         pca.components_[i, 0],
        #         pca.components_[i, 1],
        #         pca.components_[i, 2],
        #         color="r",
        #         length=1,
        #         normalize=True,
        #     )
        #     ax.text2D(
        #         pca.components_[i, 0],
        #         pca.components_[i, 1],
        #         pca.components_[i, 2],
        #         loadings.index[i],
        #         color="g",
        #         ha="center",
        #         va="center",
        #         size=10,
        #     )
        #
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.set_zlim(-1, 1)
        # ax.set_xlabel("PC1")
        # ax.set_ylabel("PC2")
        # ax.set_zlabel("PC3")
        # ax.set_title("PCA Loadings Plot")
        # plt.show(block=False)

    return pca
