import pandas as pd
import numpy as np
from aenum import MultiValueEnum


class DataType(MultiValueEnum, init="type string"):
    IDENTIFIER = (0, "identifier")
    UNINFORMATIVE = (1, "uninformative")
    UNKNOWN = (3, "unknown")

    CATEGORICAL_BINARY = (10, "categorical binary")
    CATEGORICAL_STRING = (11, "categorical string")
    CATEGORICAL_INTEGER = (12, "categorical integer")

    NUMERIC = (20, "numeric")

    @staticmethod
    def infer_dtype(
        ser: pd.Series,
        num_max_categories: int = None,
        fraction_max_categories: float = None,
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
            return DataType.CATEGORICAL_BINARY

        if "name" in ser.name.lower() or "id" in ser.name.lower() or n_uniq == n:
            return DataType.IDENTIFIER

        if ser.apply(lambda x: isinstance(x, str)).all():
            if n_uniq <= max_categories:
                return DataType.CATEGORICAL_STRING
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


def infer_data_type(
    ser: pd.Series,
    num_max_categories: int = None,
    fraction_max_categories: float = None,
):
    ser = ser.dropna()
    n = len(ser)

    # Get max categories
    if num_max_categories:
        max_categories = num_max_categories
    elif fraction_max_categories:
        max_categories = int(fraction_max_categories * n)
    else:
        max_categories = int(0.1 * n)

    uniq_vals = ser.unique()
    n_uniq = len(uniq_vals)

    print("Name of series:", ser.name)
    print("Number of uniques:", n_uniq)

    if n_uniq == 1:
        print("Inferred Type:", "uninformative feature (recommend dropping)")
        print("")
        return

    if n_uniq == 2:
        print("Inferred Type:", "binary categorical")
        print("")
        return

    if "name" in ser.name.lower() or "id" in ser.name.lower() or n_uniq == n:
        print("Inferred Type:", "identifier (recommend dropping)")
        print("")
        return

    # Check for presence of string
    if ser.apply(lambda x: isinstance(x, str)).all():
        print("Inferred Type:", "categorical string")
        print("")
        return

    # Check for mixed types
    if (
        ser.apply(lambda x: isinstance(x, str)).any()
        and ser.apply(lambda x: not isinstance(x, str)).any()
    ):
        print("Inferred Type:", "mixed type")
        print("")
        return

    # Infering as numbers
    if (ser == ser.astype(int, errors="raise")).all():
        if len(ser.unique()) <= max_categories:
            print("Inferred Type:", "categorical integer")
        else:
            print("Inferred Type:", "numeric integer")
    else:
        print("Inferred Type:", "numeric float")

    print("")
    pass


def check_data(df: pd.DataFrame):
    cols = df.columns
    n_cols = len(cols)
    n_rows = len(df)

    print("=== SIZE ===")
    print(f"DataFrame has {n_cols} columns: {cols.to_list()}")
    print(f"DataFrame has {n_rows} rows")
    print("")

    print("=== INCOMPLETENESS ===")
    cols_with_nans = [col for col in df.columns if df[col].isna().any()]
    print(
        f"There are {len(cols_with_nans)} columns containing nan values: {cols_with_nans}"
    )

    rows_with_nans = df.isna().any(axis=1).sum()
    print(f"There are {rows_with_nans} rows containing nan values")

    nans = df.isna().sum().sum()
    print(f"There are a total of {nans} nan values")
    print("")

    print("=== DATA TYPES ===")

    # Create subdf for data types

    for col in cols:
        dtype = DataType.infer_dtype(df[col])

        feature_str = ""
        match dtype:
            case DataType.CATEGORICAL_BINARY:
                pass

        print(f"Column {col} is of type {dtype.string}")

    data_types_df = pd.DataFrame(
        {
            "column name": cols,
            "dtype": [DataType.infer_dtype(df[col]).string for col in cols],
        }
    )
    "asdasdad"
    print(data_types_df)

    pass
