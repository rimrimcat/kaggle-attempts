import pandas as pd
from sklearn.impute import SimpleImputer

from __scripts__.data import ColumnTransformer, DataType, clean_data

df = pd.read_csv("titanic/train.csv")

df = clean_data(df, drop_missing=False, drop_uninformative=True)

dtype_dict = DataType.infer_df_dtype(df)
dtype_dict["Survived"] = dtype_dict["Survived"].set_ordinal()

# check_corr(ddf, "Survived", dtype_dict)

X_trans, Y_trans = ColumnTransformer.create_transformers(
    df,
    dtype_dict,
    labels="Survived",
)
X_scaled, I_trans = X_trans.fit_transform(df, imputer=SimpleImputer())
X_revert = X_trans.inverse_transform_df(X_scaled)


# print(X_revert)
# summarize_data(X_revert)
# print(X_scaled)
