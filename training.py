import os
import pickle
import pandas as pd

try:
    from prediction import train_classifier, train_regressor
except ImportError:
    # หากฟังก์ชันเทรนอยู่ในไฟล์ชื่ออื่น สามารถเปลี่ยนคำว่า prediction เป็นชื่อไฟล์นั้นได้เลยครับ
    raise ModuleNotFoundError("หาฟังก์ชันเทรนไม่พบ กรุณาตรวจสอบว่าฟังก์ชัน train_classifier อยู่ในไฟล์ไหนในโฟลเดอร์ของคุณ")

if __name__ == "__main__":
    print("⏳ กำลังเริ่มดาวน์โหลดข้อมูลและทดสอบระบบการเทรนบนเครื่องคอมพิวเตอร์...")

    # ดึงฟังก์ชันจัดการข้อมูลจาก app.py มาใช้งานทดสอบ
    try:
        from app import build_data
        df = build_data()
        print("✅ โหลดและรวมตารางข้อมูลสำเร็จ ขนาดข้อมูลหลัก:", df.shape)
    except Exception as e:
        print("❌ เกิดข้อผิดพลาดขณะโหลดข้อมูลจาก app.py:", e)
        exit(1)

    # 1. ทดสอบโครงสร้างและการเทรน Classifier พร้อมสั่งบันทึกไฟล์เซฟโมเดล
    print("\n--- 1. Testing Random Forest Classifier ---")
    clf, acc, clf_cols = train_classifier(df)
    print(f"📈 Model Accuracy Score: {acc * 100:.2f}%")
    print(f"📊 Features used count: {len(clf_cols)}")
    
    # สั่งบันทึกไฟล์ผลลัพธ์ลงเครื่องคอมพิวเตอร์ของคุณ
    with open("classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("clf_cols.pkl", "wb") as f:
        pickle.dump(clf_cols, f)
    print("💾 บันทึกไฟล์ classifier.pkl และ clf_cols.pkl เรียบร้อย!")

    # 2. ทดสอบโครงสร้างและการเทรน Regressor พร้อมสั่งบันทึกไฟล์เซฟโมเดล
    print("\n--- 2. Testing Random Forest Regressor ---")
    reg, mae, rmse, reg_cols = train_regressor(df)
    print(f"📉 Model MAE (Mean Absolute Error): ${mae:,.2f}")
    print(f"📉 Model RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    print(f"📊 Features used count: {len(reg_cols)}")
    
    # สั่งบันทึกไฟล์ผลลัพธ์ลงเครื่องคอมพิวเตอร์ของคุณ
    with open("regressor.pkl", "wb") as f:
        pickle.dump(reg, f)
    with open("reg_cols.pkl", "wb") as f:
        pickle.dump(reg_cols, f)
    print("💾 บันทึกไฟล์ regressor.pkl และ reg_cols.pkl เรียบร้อย!")

    print("\n🎉 ระบบโมเดล Machine Learning ทำงานได้สมบูรณ์แบบ 100% และเซฟไฟล์เรียบร้อย!")