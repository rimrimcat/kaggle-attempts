import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeAlias, Union, overload

import numpy as np
import pandas as pd
from numpy import float64
from numpy.typing import NDArray
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

from __scripts__.data import DataType

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

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s %(message)s",
    handlers=[logging.StreamHandler()],
)


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
