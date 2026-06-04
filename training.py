import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def train_classifier(df):

    customer = (
        df.groupby("CustomerKey")
        .agg({
            "Sales Amount":"sum",
            "Country":"first"
        })
        .reset_index()
    )

    median_sales = customer["Sales Amount"].median()

    customer["Buy"] = (
        customer["Sales Amount"] > median_sales
    ).astype(int)

    X = pd.get_dummies(
        customer[["Country"]],
        drop_first=True
    )

    y = customer["Buy"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    clf.fit(X_train,y_train)

    pred = clf.predict(X_test)

    acc = accuracy_score(y_test,pred)

    return clf,acc,X.columns


def train_regressor(df):

    X = pd.get_dummies(
        df[
            [
                "Country",
                "Category",
                "List Price",
                "Order Quantity"
            ]
        ],
        drop_first=True
    )

    y = df["Sales Amount"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    reg.fit(X_train,y_train)

    pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test,pred)

    rmse = np.sqrt(
        mean_squared_error(y_test,pred)
    )

    return reg,mae,rmse,X.columns