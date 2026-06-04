import pandas as pd

def predict_buy(model, feature_columns, country):
    sample = pd.DataFrame({
        "Country": [country]
    })

    sample = pd.get_dummies(sample, drop_first=True)
    sample = sample.reindex(columns=feature_columns, fill_value=0)

    pred = model.predict(sample)
    return "BUY" if pred[0] == 1 else "NOT BUY"


def predict_sales(model, feature_columns, country, category, price, qty):
    sample = pd.DataFrame({
        "Country": [country],
        "Category": [category],
        "List Price": [price],
        "Order Quantity": [qty]
    })

    sample = pd.get_dummies(sample, drop_first=True)
    sample = sample.reindex(columns=feature_columns, fill_value=0)

    return model.predict(sample)[0]