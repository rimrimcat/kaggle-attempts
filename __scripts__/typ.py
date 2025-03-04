import logging
from collections import UserDict
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Literal,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
from numpy.matlib import datetime64
from pandas import Categorical
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
from typing_extensions import override


class BasicDataType(Enum):
    OTHER = auto()
    NOMINAL = auto()
    ORDINAL = auto()
    CONTINUOUS = auto()

    def is_categorical(self):
        return (self == BasicDataType.NOMINAL) or (self == BasicDataType.ORDINAL)

    def is_discrete(self):
        return (self == BasicDataType.NOMINAL) or (self == BasicDataType.ORDINAL)

    def is_nominal(self):
        return self == BasicDataType.NOMINAL

    def is_ordinal(self):
        return (self == BasicDataType.ORDINAL) or (self == BasicDataType.CONTINUOUS)

    def is_continuous(self):
        return self == BasicDataType.CONTINUOUS


class DataTypeElement:
    __slots__ = ("_value_", "should_drop")
    _value_: int
    should_drop: bool
    basic_type: BasicDataType


class DataTypeDict(UserDict[str, "DataType"]):
    def set_ordinal(self, col: str):
        self.data[col] = self.data[col].get_ordinal()


class DataType(DataTypeElement, Enum):
    IDENTIFIER = auto(), True, BasicDataType.OTHER
    UNINFORMATIVE = auto(), True, BasicDataType.OTHER
    UNKNOWN = auto(), True, BasicDataType.OTHER
    TIME_DATA = auto(), False, BasicDataType.OTHER

    CATEGORICAL_NOMINAL_BINARY = auto(), False, BasicDataType.NOMINAL
    CATEGORICAL_ORDINAL_BINARY = auto(), False, BasicDataType.ORDINAL

    CATEGORICAL_NOMINAL_STRING = auto(), False, BasicDataType.NOMINAL
    CATEGORICAL_ORDINAL_STRING = auto(), False, BasicDataType.ORDINAL

    CATEGORICAL_INTEGER = auto(), False, BasicDataType.ORDINAL

    NUMERIC = auto(), False, BasicDataType.CONTINUOUS

    def __new__(cls, value: int, should_drop: bool, basic_type: BasicDataType):
        obj = DataTypeElement.__new__(cls)
        obj._value_ = value
        obj.should_drop = should_drop
        obj.basic_type = basic_type
        return obj

    def __bool__(self):
        return not self.should_drop

    def is_binary(self):
        return "BINARY" in self.name

    def is_categorical(self) -> bool:
        return self.basic_type.is_categorical()

    def is_discrete(self):
        return self.basic_type.is_discrete()

    def is_ordinal(self):
        return self.basic_type.is_ordinal()

    def is_continuous(self):
        return self.basic_type.is_continuous()

    def get_ordinal(self):
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

        # detect unsortable datetime object
        if pd.api.types.is_datetime64_any_dtype(ser.dtype):
            return DataType.TIME_DATA

        title = str(ser.name)
        n = len(ser)
        uniq_vals = ser.unique()
        n_uniq = len(uniq_vals)

        # get max categories
        if num_max_categories:
            max_categories = num_max_categories
        elif fraction_max_categories:
            max_categories = int(fraction_max_categories * n)
        else:
            max_categories = int(0.1 * n)

        if n_uniq == 1:
            return DataType.UNINFORMATIVE

        # binary data types
        if n_uniq == 2:
            uniq_vals_set = set(uniq_vals.tolist())
            if uniq_vals_set == {True, False}:
                return DataType.CATEGORICAL_ORDINAL_BINARY
            elif uniq_vals_set == {0, 1}:
                return DataType.CATEGORICAL_ORDINAL_BINARY

            return DataType.CATEGORICAL_NOMINAL_BINARY

        if "name" in title.lower() or title.lower() == "id" or n_uniq == n:
            return DataType.IDENTIFIER

        # handle special case of pandas Categorical
        if isinstance(uniq_vals, Categorical):
            return DataType.CATEGORICAL_ORDINAL_BINARY

        # differentiate between categorical string and uninformative
        if ser.apply(lambda x: isinstance(x, str)).all():
            if n_uniq <= max_categories:
                return DataType.CATEGORICAL_NOMINAL_STRING
            else:
                return DataType.UNINFORMATIVE

        # differentiate between categorical int and continuous
        if (ser == ser.astype(int, errors="raise")).all():
            uniq_vals.sort()
            if n_uniq <= max_categories and (np.diff(uniq_vals) == 1).all():
                return DataType.CATEGORICAL_INTEGER
            else:
                return DataType.NUMERIC
        else:
            return DataType.NUMERIC

        return DataType.UNKNOWN

    @staticmethod
    def infer_df_dtype(df: pd.DataFrame) -> "DataTypeDict":
        logger = logging.getLogger("types.DataType.infer_df_dtype")

        dtype_dict = {col: DataType.infer_ser_dtype(df[col]) for col in df.columns}

        for k, v in dtype_dict.items():
            if v.is_categorical() and not v.is_ordinal():
                logger.info(
                    f"NOTE: Column '{k}' ({v.name}) has unknown ordinality. If the column is ordinal, set it by `dtype_dict.set_ordinal('{k}')`"
                )

        df.attrs = {"dtype_dict": DataTypeDict(dtype_dict)}

        return df.attrs["dtype_dict"]


@dataclass(slots=True, frozen=True)
class Task:
    multi_label: bool
    task: Literal["regression", "classification"]
    time_series: bool = False
    labels: Optional[list[str]] = None

    def get_scorer(self):
        match self.task:
            case "regression":
                return make_scorer(mean_squared_error, greater_is_better=False)
            case "classification":
                return make_scorer(accuracy_score)
        raise NotImplementedError(f"Unknown task {self.task}")

    @staticmethod
    def infer_from_df(df: pd.DataFrame, labels: Union[str, list[str]]):
        dtype_dict = df.attrs["dtype_dict"]

        if isinstance(labels, str):
            labels = [labels]

        # Check if there is a column with time_series

        return Task(
            multi_label=True if len(labels) > 1 else False,
            task=(
                "classification"
                if dtype_dict[labels[0]].is_categorical()
                else "regression"
            ),
            time_series=(
                True
                if any([dtype_dict[col] == DataType.TIME_DATA for col in df.columns])
                else False
            ),
            labels=labels,
        )

    @staticmethod
    def regr_ml():
        return Task(True, "regression")

    @staticmethod
    def regr_sl():
        return Task(False, "regression")

    @staticmethod
    def clf_ml():
        return Task(True, "classification")

    @staticmethod
    def clf_sl():
        return Task(False, "classification")


@dataclass(slots=True, frozen=True)
class StatResult:
    symbol: str
    value: float
    pvalue: Optional[float]

    @property
    def strength(self):
        abs_val = abs(self.value)

        if abs_val < 0.1:
            return "Negligible"
        elif abs_val < 0.3:
            return "Weak"
        elif abs_val < 0.5:
            return "Fair"
        elif abs_val < 0.7:
            return "Moderate"
        elif abs_val < 1:
            return "Very Strong"
        elif abs_val == 1:
            return "Perfect"

    def __str__(self):
        if self.pvalue:
            pval = f"{self.pvalue:.1e}".replace("e", r"\mathrm{e}")

            if self.pvalue < 0.05:
                return f"${self.symbol}={self.value:.2f}, *p={pval}$"

            return f"${self.symbol}={self.value:.2f}, p={pval}$"

        else:
            return f"${self.symbol}={self.value:.2f}$"
