try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    pass

import os
from math import log

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from __scripts__.data import (
    DataFrameTransformer,
    DataType,
    FunctionInFunction,
    check_corr,
    clean_data,
    ColumnTransformer,
)

load_dotenv(".env")
DATABASE_URL = os.environ.get("DATABASE_URL")


def extract_common_titles(df: pd.Series):
    title_val_counts = (
        df.apply(lambda name: name.split(",")[1].split(".")[0])
        .value_counts()
        .apply(log)
    )
    title_val_counts.sort_values(ascending=False)

    id_nl = np.diff(title_val_counts * -1).argmax()
    x = np.mean(title_val_counts.iloc[[id_nl, id_nl + 1]])

    common_titles = set(title_val_counts[title_val_counts >= x].index)

    def extract_title(name: str):
        title = name.split(",")[1].split(".")[0]
        return title if title in common_titles else "Other"

    return extract_title


def has_nickname(name: str):
    return '"' in name


def has_other_name(name: str):
    return "(" in name


def get_cabin_letter(cabin: str):
    try:
        return cabin[0]
    except:
        return cabin


def get_ticket_number(ticket: str):
    try:
        return int(ticket.split(" ")[-1])
    except:
        return np.nan


F_Engg = DataFrameTransformer(
    [
        # (FunctionInFunction(extract_common_titles), "Name", "Title"),
        (has_nickname, "Name", "HasNickname"),
        (has_other_name, "Name", "HasOthername"),
        (get_cabin_letter, "Cabin", "CabinLetter"),
        (get_ticket_number, "Ticket", "TicketNo"),
        (lambda ser: ser.sum(axis=1), ["SibSp", "Parch"], "FamilySize"),
        (lambda ser: ser == 0, ["FamilySize"], "IsAlone"),
        (lambda ser: ser.notna(), ["Cabin"], "HasCabin"),
        ("drop", "TicketNo"),
    ]
)


df = pd.read_csv("titanic/train.csv")
df = F_Engg.fit_transform(df)


df = clean_data(
    df,
    drop_missing_rows=False,
    drop_uninformative=True,
    add_new_features=False,
)
# F_Engg.transformers += new_tforms


dtype_dict = DataType.infer_df_dtype(df)
# Perform modifications here

check_corr(df, "Survived", 4)
raise ValueError()

Trans_X, Trans_Y, task = ColumnTransformer.create_transformers(
    df,
    labels="Survived",
)


X_scaled = Trans_X.fit_transform(df, simple_imputer=SimpleImputer())
Y_scaled = Trans_Y.fit_transform(df)

do_pca(X_scaled, Y_scaled, Trans_X, Trans_Y, task=task)

raise ValueError()

# opt_base_model(X_scaled, Y_scaled, task=task)


# # # RandomForestClassifier got the best score

# # Try LDA
# best_mdl = tune_model(
#     X_scaled,
#     Y_scaled,
#     task=task,
#     model_cls=LinearDiscriminantAnalysis,
#     param_func=partial(
#         ModelParams.lda, task=task, n_features=X_scaled.shape[1], n_classes=2
#     ),
#     total_trials=50,
# )


# best_mdl = tune_model(
#     X_scaled,
#     Y_scaled,
#     task=task,
#     model_cls=RandomForestClassifier,
#     param_func=partial(ModelParams.random_forest, task=task),
#     study_name="TITANIC",
#     storage=DATABASE_URL,
#     total_trials=500,
# )

# TEST

test_df = pd.read_csv("titanic/test.csv")
test_df = F_Engg.transform(test_df)

X_test_scaled = Trans_X.transform(test_df)
Y_test_pred = best_mdl.predict(X_test_scaled)
Y_test_pred = Y_test_pred.astype(int).ravel()

test_df["Survived"] = Y_test_pred
test_df[["PassengerId", "Survived"]].to_csv("titanic/answer.csv", index=False)
