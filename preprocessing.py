# ให้มีแค่นี้พอครับ ห้ามมีบรรทัดที่แอบไป import preprocessing ซ้ำเด็ดขาด!
import pandas as pd

def clean_currency(df):
    cols = [
        "Unit Price",
        "Extended Amount",
        "Product Standard Cost",
        "Total Product Cost",
        "Sales Amount",
        "List Price",
        "Standard Cost"
    ]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col]
                .astype(str)
                .str.replace(r"[$,]", "", regex=True),
                errors="coerce"
            )
    return df


def remove_duplicates(df):
    return df.drop_duplicates()


def remove_missing(df):
    return df.dropna()


def remove_outliers(df):
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[
            (df[col] >= lower)
            &
            (df[col] <= upper)
        ]
    return df


def clean_data(df):
    df = remove_duplicates(df)
    df = remove_missing(df)
    df = remove_outliers(df)
    return df