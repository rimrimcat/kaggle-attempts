import pandas as pd

from __scripts__.data import check_data, check_corr, DataType

df = pd.read_csv("titanic/train.csv")


ddf = check_data(df)


dtype_dict = DataType.infer_df_dtype(ddf)
dtype_dict["Survived"] = dtype_dict["Survived"].set_ordinal()


check_corr(ddf, "Survived", dtype_dict)
