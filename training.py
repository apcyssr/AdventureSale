import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


def train_classifier(df):
    customer = (
        df.groupby("CustomerKey")
        .agg({
            "Sales Amount": "sum",
            "Country": "first"
        })
        .reset_index()
    )

    median_sales = customer["Sales Amount"].median()
    customer["Buy"] = (customer["Sales Amount"] > median_sales).astype(int)

    X = pd.get_dummies(customer[["Country"]], drop_first=True)
    y = customer["Buy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ปรับลดความซับซ้อนเพื่อประหยัด RAM บน Render
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    return clf, acc, X.columns


def train_regressor(df):
    X = pd.get_dummies(
        df[["Country", "Category", "List Price", "Order Quantity"]],
        drop_first=True
    )
    y = df["Sales Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ปรับลดความซับซ้อนเพื่อประหยัด RAM บน Render
    reg = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    return reg, mae, rmse, X.columns


# ==========================================
# 🧪 สั่งเจนไฟล์โมเดลสำเร็จรูปออกมาใช้งาน (.pkl)
# ==========================================
if __name__ == "__main__":
    print("⏳ กำลังโหลดข้อมูลและสร้างไฟล์โมเดลสำเร็จรูป...")
    
    try:
        from app import build_data
        df = build_data()
        print("✅ ดึงข้อมูลสำเร็จขนาด:", df.shape)
    except Exception as e:
        print("❌ ดึงข้อมูลไม่สำเร็จ:", e)
        exit(1)

    # คำสั่งสร้างโฟลเดอร์ models บนเครื่องคอมพิวเตอร์ของคุณ
    os.makedirs("models", exist_ok=True)

    # 1. บันทึก Classifier
    clf, acc, clf_cols = train_classifier(df)
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("models/clf_cols.pkl", "wb") as f:
        pickle.dump(list(clf_cols), f)

    # 2. บันทึก Regressor
    reg, mae, rmse, reg_cols = train_regressor(df)
    with open("models/regressor.pkl", "wb") as f:
        pickle.dump(reg, f)
    with open("models/reg_cols.pkl", "wb") as f:
        pickle.dump(list(reg_cols), f)
    
    print("\n🎉 บันทึกไฟล์ป้อนข้อมูลสำเร็จ! คุณจะพบโฟลเดอร์ชื่อ 'models' ในโปรเจกต์ของคุณแล้วครับ")