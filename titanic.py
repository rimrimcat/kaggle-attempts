import pandas as pd
from sklearn.impute import SimpleImputer

from __scripts__.data import DataType, clean_data
from __scripts__.model import ColumnTransformer

df = pd.read_csv("titanic/train.csv")

ddf = clean_data(df, drop_missing=False, drop_uninformative=True)

dtype_dict = DataType.infer_df_dtype(ddf)
dtype_dict["Survived"] = dtype_dict["Survived"].set_ordinal()

# check_corr(ddf, "Survived", dtype_dict)

X_trans, Y_trans = ColumnTransformer.create_transformers(
    ddf,
    dtype_dict,
    labels="Survived",
)
X_scaled, I_trans = X_trans.fit_transform(ddf, imputer=SimpleImputer())
X_revert = X_trans.inverse_transform_df(X_scaled)


# print(X_revert)
# summarize_data(X_revert)
# print(X_scaled)
