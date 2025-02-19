import logging
from enum import Enum, auto
from math import ceil
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeAlias, Union, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import float64
from numpy.typing import NDArray
from pandas.core.frame import DataFrame
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.impute._base import _BaseImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.preprocessing._encoders import _BaseEncoder

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
transform_list: TypeAlias = list[tuple[str, transform_item, Union[list[str], slice]]]


class DataTypeElement:
    __slots__ = ("_value_", "should_drop")
    _value_: int
    should_drop: bool


class DataType(DataTypeElement, Enum):
    IDENTIFIER = auto(), True
    UNINFORMATIVE = auto(), True
    UNKNOWN = auto(), True

    CATEGORICAL_NOMINAL_BINARY = auto(), False
    CATEGORICAL_ORDINAL_BINARY = auto(), False

    CATEGORICAL_NOMINAL_STRING = auto(), False
    CATEGORICAL_ORDINAL_STRING = auto(), False

    CATEGORICAL_INTEGER = auto(), False

    NUMERIC = auto(), False

    def __new__(cls, value: int, should_drop: bool):
        obj = DataTypeElement.__new__(cls)
        obj._value_ = value
        obj.should_drop = should_drop
        return obj

    def __bool__(self):
        return not self.should_drop

    def is_categorical(self) -> bool:
        return "CATEGORICAL" in self.name

    def is_ordinal(self):
        return "ORDINAL" in self.name or "INTEGER" in self.name

    def is_binary(self):
        return "BINARY" in self.name

    def is_continuous(self):
        return "NUMERIC" in self.name

    def set_ordinal(self):
        if "NOMINAL" in self.name:
            return DataType(self._value_ + 1)
        else:
            return self

    @staticmethod
    def infer_ser_dtype(
        ser: pd.Series,
        num_max_categories: Optional[int] = None,
        fraction_max_categories: Optional[float] = None,
    ) -> "DataType":
        ser = ser.dropna()
        n = len(ser)
        uniq_vals = ser.unique()
        n_uniq = len(uniq_vals)

        # Get max categories
        if num_max_categories:
            max_categories = num_max_categories
        elif fraction_max_categories:
            max_categories = int(fraction_max_categories * n)
        else:
            max_categories = int(0.1 * n)

        if n_uniq == 1:
            return DataType.UNINFORMATIVE

        if n_uniq == 2:
            return DataType.CATEGORICAL_NOMINAL_BINARY

        if "name" in ser.name.lower() or "id" in ser.name.lower() or n_uniq == n:
            return DataType.IDENTIFIER

        if ser.apply(lambda x: isinstance(x, str)).all():
            if n_uniq <= max_categories:
                return DataType.CATEGORICAL_NOMINAL_STRING
            else:
                return DataType.UNINFORMATIVE

        if (ser == ser.astype(int, errors="raise")).all():
            if n_uniq <= max_categories:
                return DataType.CATEGORICAL_INTEGER
            else:
                return DataType.NUMERIC
        else:
            return DataType.NUMERIC

        return DataType.UNKNOWN

    @staticmethod
    def infer_df_dtype(df: pd.DataFrame) -> dict[str, "DataType"]:
        logger = logging.getLogger("data.DataType.infer_df_dtype")

        dtype_dict = {col: DataType.infer_ser_dtype(df[col]) for col in df.columns}

        for k, v in dtype_dict.items():
            if v.is_categorical() and not v.is_ordinal():
                logger.info(
                    f"NOTE: Column '{k}' ({v.name}) has unknown ordinality. If the column is ordinal, set it by `dtype_dict['{k}'] = dtype_dict['{k}'].set_ordinal()`"
                )

        return dtype_dict


class ColumnTransformer(_ColumnTransformer):
    """
    InverseColumnTransformer that handles drop and passthrough

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
    def _create_column_imputer(
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

                def onehot_to_ordinal(X):
                    return np.where(
                        (X == 0).all(axis=1)[:, None],
                        np.nan,
                        np.argmax(X, axis=1)[:, None],
                    )

                def make_fun(num_categories: int):
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

                fw_trans = FunctionTransformer(onehot_to_ordinal)
                rev_trans = FunctionTransformer(make_fun(num_categories))

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

        return ColumnTransformer(
            transformers=transformers
        )  # does not have inverse transform

    @staticmethod
    def create_transformers(
        df: pd.DataFrame,
        dtype_dict: dict[str, DataType],
        labels: Union[str, list[str]],
        trans_dict: Optional[dict[str, Union[BaseEstimator, _BaseEncoder]]] = None,
    ) -> tuple["ColumnTransformer", "ColumnTransformer"]:
        if isinstance(labels, str):
            labels = [labels]

        x_cols = [col for col in df.columns if col not in labels]
        df_X = df[x_cols]
        df_Y = df[labels]

        trans_X = ColumnTransformer._create_column_transformer(
            df_X, dtype_dict, trans_dict
        )
        trans_Y = ColumnTransformer._create_column_transformer(
            df_Y, dtype_dict, trans_dict
        )

        return (trans_X, trans_Y)

    @overload
    def fit_transform(
        self: Any,
        X: Any,
        y: Any = None,
        imputer: Union[
            BaseEstimator, _BaseImputer, dict[str, Union[BaseEstimator, _BaseImputer]]
        ] = None,
        **fit_params,
    ) -> tuple[_X_t, "ColumnTransformer"]: ...

    @overload
    def fit_transform(
        self: Any,
        X: Any,
        y: Any = None,
        imputer: Literal[None] = None,
        **fit_params,
    ) -> _X_t: ...

    def fit_transform(
        self, X, y=None, imputer=None, **fit_params
    ) -> Union[tuple[_X_t, "ColumnTransformer"], _X_t]:
        X_t = super().fit_transform(X, y, **fit_params)

        if imputer:
            # Create trans_dict
            if isinstance(imputer, _BaseImputer) or isinstance(imputer, BaseEstimator):
                trans_dict = {k: clone(imputer) for k in self.output_indices_.keys()}
            elif isinstance(imputer, dict):
                trans_dict = imputer
            else:
                raise ValueError(f"Unknown type for imputer: {type(imputer)}")

            trans_imputer = ColumnTransformer._create_column_imputer(
                trans_dict,
                self.output_indices_,
                X_t,
            )

            X_t = trans_imputer.fit_transform(X_t)

            return (X_t, trans_imputer)
        return X_t


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
            case DataType.CATEGORICAL_NOMINAL_BINARY:
                value = f"{uv[0]}, {uv[1]}"
            case DataType.CATEGORICAL_NOMINAL_STRING:
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
                value = f"({len(uv)} unique values) {uv[0]}, {uv[1]}, {uv[2]}, ..., {uv[-2]}, {uv[-1]}"
                if len(value) > 50:
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
    drop_missing: bool = False,
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

    summary_0 = summarize_data(df, print_summary=False)

    if drop_uninformative:
        df = df[[col for dt, col in zip(summary_0["data_types"], df.columns) if dt]]

    if drop_missing:
        df = df.dropna()

    summary_1 = summarize_data(df, print_summary=False)

    if print_summary:
        if drop_uninformative:
            mod_data_types_df: pd.DataFrame = summary_0["data_types_df"]
            mod_data_types_df["Notes"] = [
                "DROPPED" if d.should_drop else "" for d in summary_0["data_types"]
            ]

            i2col = {
                str(k): v
                for k, v in summary_0["data_types_df"].to_dict()["Column"].items()
            }
            col2newi = {
                v: str(k)
                for k, v in summary_1["data_types_df"].to_dict()["Column"].items()
            }

            new_indices = [
                "" if d.should_drop else col2newi[i2col[ind]]
                for ind, d in zip(
                    mod_data_types_df.index.astype(str), summary_0["data_types"]
                )
            ]
            mod_data_types_df.index = new_indices

        print("=#=#=#=#=#=#=#=#= SUMMARY =#=#=#=#=#=#=#=#=")

        print("=== SIZE ===")
        print(f"{summary_0['n_cols']} -> {summary_1['n_cols']} columns")
        print(f"{summary_0['n_rows']} -> {summary_1['n_rows']} rows")
        print("")

        print("=== INCOMPLETENESS ===")
        nan_col_str = f": {summary_1['nan_cols']}" if summary_1["nan_cols"] else ""
        print(
            f"{len(summary_0['nan_cols'])} -> {len(summary_1['nan_cols'])} columns containing NaNs{nan_col_str}"
        )
        print(
            f"{summary_0['nan_rows']} -> {summary_1['nan_rows']} rows containing NaNs"
        )
        print(f"{summary_0['nans']} -> {summary_1['nans']} total NaNs")
        print("")

        print("=== DATA TYPES ===")
        print(summary_0["data_types_df"].to_string())
        print("")

    if summary_0["nans"]:
        logger = logging.getLogger("data.clean_data")
        logger.info(
            "NOTE: NaN values may be present in data. Either drop missing values or specify an imputer for the Column Transformer."
        )

    return df


def plot_monovariate_violin(
    df: pd.DataFrame,
    dtype_dict: dict,
    num_plots_x: int,
    num_plots_y: int,
):
    fig, axes = plt.subplots(num_plots_y, num_plots_x)
    fig.tight_layout()
    axes = axes.flatten()

    for i, x in enumerate(df.columns.to_list()):
        if dtype_dict[x].is_categorical():
            filter_df = df[x].dropna()
            sns.violinplot(x=filter_df.astype(str), ax=axes[i])
        else:
            filter_df = df[x].dropna()
            sns.violinplot(x=filter_df, ax=axes[i])

    n_diff = num_plots_x * num_plots_y - len(df.columns)
    if n_diff:
        for i in range(n_diff):
            fig.delaxes(axes[-(i + 1)])

    plt.suptitle("Feature Violin Plots")
    plt.show(block=False)


def plot_bivariate_violin(
    df: pd.DataFrame,
    dtype_dict: dict,
    x_cols: list[str],
    y: str,
    num_plots_x: int,
    num_plots_y: int,
):
    fig, axes = plt.subplots(num_plots_y, num_plots_x)
    fig.tight_layout()
    axes = axes.flatten()

    for i, x in enumerate(x_cols):
        filter_df = df[[x, y]].dropna()
        corr = None

        match dtype_dict[x].is_categorical(), dtype_dict[y].is_categorical():
            case True, True:
                sns.violinplot(
                    x=filter_df[x].astype(str),
                    y=filter_df[y].astype(str),
                    ax=axes[i],
                )

                if dtype_dict[x].is_ordinal() and dtype_dict[y].is_ordinal():
                    corrdf = filter_df.copy()
                    corrdf[x] = pd.factorize(filter_df[x], sort=True)[0]
                    corrdf[y] = pd.factorize(filter_df[y], sort=True)[0]
                    corr = corrdf.corr()[x][y]
            case False, True:
                sns.violinplot(
                    x=filter_df[x],
                    y=filter_df[y].astype(str),
                    ax=axes[i],
                )

                if dtype_dict[y].is_ordinal():
                    corrdf = filter_df.copy()
                    corrdf[y] = pd.factorize(filter_df[y], sort=True)[0]
                    corr = corrdf.corr()[x][y]

            case True, False:
                sns.violinplot(
                    x=filter_df[x].astype(str),
                    y=filter_df[y],
                    ax=axes[i],
                )

                if dtype_dict[x].is_ordinal():
                    corrdf = filter_df.copy()
                    corrdf[x] = pd.factorize(filter_df[x], sort=True)[0]
                    corr = corrdf.corr()[x][y]

            case False, False:
                sns.violinplot(
                    x=filter_df[x],
                    y=filter_df[y],
                    ax=axes[i],
                )

                corr = filter_df.corr()[x][y]

            case _, _:
                raise NotImplementedError("Not implemented yet!")

        axes[i].invert_yaxis()
        if corr:
            axes[i].text(
                1.0,
                1.0,
                f"CORR: {round(corr, 4)}",
                ha="right",
                va="top",
                transform=axes[i].transAxes,
                bbox=dict(facecolor="white", alpha=0.5),
            )

    n_diff = num_plots_x * num_plots_y - len(x_cols)
    if n_diff:
        for i in range(n_diff):
            fig.delaxes(axes[-(i + 1)])

    plt.suptitle("Bivariate Violin Plots")
    plt.show(block=False)


def check_corr(
    df: pd.DataFrame, label: str, dtype_dict: dict[str, DataType], num_plots_x: int = 3
):
    x_cols = df.columns.drop(label).to_list()
    y = label

    n_plots = len(x_cols)

    plot_monovariate_violin(
        df, dtype_dict, num_plots_x, ceil((n_plots + 1) / num_plots_x)
    )

    plot_bivariate_violin(
        df, dtype_dict, x_cols, y, num_plots_x, ceil(n_plots / num_plots_x)
    )

    input("Press Enter to continue.")
    pass
