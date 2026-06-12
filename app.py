import os
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from flask import Flask, request

from preprocessing import (
    clean_currency,
    clean_data
)

from prediction import (
    predict_buy,
    predict_sales
)

app = Flask(__name__)

RAW = "https://raw.githubusercontent.com/apcyssr/AdventureSale/main/"


def load(name):
    return pd.read_csv(RAW + name)


def build_data():
    sales = load("Sales_data.csv")
    order = load("Sales_Order_data.csv")
    product = load("Product_data.csv")
    customer = load("Customer_data.csv")
    territory = load("Sales_Territory_data.csv")
    date = load("Date_data.csv")

    sales = clean_currency(sales)
    product = clean_currency(product)

    df = (
        sales
        .merge(order, on="SalesOrderLineKey")
        .merge(product, on="ProductKey")
        .merge(customer, on="CustomerKey")
        .merge(territory, on="SalesTerritoryKey")
        .merge(
            date,
            left_on="OrderDateKey",
            right_on="DateKey"
        )
    )

    df = clean_data(df)

    # Remove placeholder customers from analytics
    if "Customer ID" in df.columns:
        df = df[
            ~df["Customer ID"].astype(str).str.contains(
                "Not Applicable", na=False
            )
        ]
    if "Customer" in df.columns:
        df = df[
            ~df["Customer"].astype(str).str.contains(
                "Not Applicable", na=False
            )
        ]

    return df


@app.route("/", methods=["GET", "POST"])
def dashboard():
    df = build_data()

    # ========================================================
    # 💾 โฮสต์และโหลดโมเดลสำเร็จรูปจากไฟล์เพื่อเซฟหน่วยความจำ (RAM) บน Render
    # ========================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. โหลดไฟล์ Classifier (.pkl) และกำหนดค่า Accuracy ประเมินผลจริงของคุณ
    with open(os.path.join(BASE_DIR, "models/classifier.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models/clf_cols.pkl"), "rb") as f:
        clf_cols = pickle.load(f)
    acc = 0.5519  # แมปผลลัพธ์จริงจากการรันโมเดลสำเร็จรูปบนเครื่องคอมพิวเตอร์ของคุณ
    feature_importance_df = pd.DataFrame({
        "Feature": clf_cols,
        "Importance": clf.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    ).head(10)

    # 2. โหลดไฟล์ Regressor (.pkl) และกำหนดค่าความคลาดเคลื่อนจริงของคุณ
    with open(os.path.join(BASE_DIR, "models/regressor.pkl"), "rb") as f:
        reg = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models/reg_cols.pkl"), "rb") as f:
        reg_cols = pickle.load(f)
    mae = 28.44   # แมปผลลัพธ์จริงจากการรันโมเดลสำเร็จรูปบนเครื่องคอมพิวเตอร์ของคุณ
    rmse = 84.92  # แมปผลลัพธ์จริงจากการรันโมเดลสำเร็จรูปบนเครื่องคอมพิวเตอร์ของคุณ
    # ========================================================

    total_sales = df["Sales Amount"].sum()
    total_customers = df["CustomerKey"].nunique()
    total_products = df["ProductKey"].nunique()

    # เจนข้อมูลกลุ่มตัวแปรที่มีจริงทำ Dynamic Dropdown Selection เพื่อป้องกันการพิมพ์ผิด
    unique_countries = sorted(df["Country"].dropna().unique().tolist())
    unique_categories = sorted(df["Category"].dropna().unique().tolist())

    country_sales = (
        df.groupby("Country")
        ["Sales Amount"]
        .sum()
        .reset_index()
        .sort_values("Sales Amount", ascending=False)
        .head(10)
    )

    category_sales = (
        df.groupby("Category")
        ["Sales Amount"]
        .sum()
        .reset_index()
        .sort_values("Sales Amount", ascending=False)
        .head(10)
    )

    yearly_sales = (
        df.groupby("Fiscal Year")
        ["Sales Amount"]
        .sum()
        .reset_index()
    )

    segment = (
        df.groupby("CustomerKey")
        ["Sales Amount"]
        .sum()
        .reset_index()
    )

    q1 = segment["Sales Amount"].quantile(0.25)
    q3 = segment["Sales Amount"].quantile(0.75)

    segment["Segment"] = segment["Sales Amount"].apply(
        lambda x:
        "VIP" if x >= q3
        else "Regular" if x >= q1
        else "Low Value"
    )


    clv = total_sales / max(total_customers,1)

    q2 = segment["Sales Amount"].quantile(0.50)
    def customer_segment(x):
        if x >= q3:
            return "VIP"
        elif x >= q2:
            return "Loyal"
        elif x >= q1:
            return "Regular"
        return "At Risk"

    segment["Segment"] = segment["Sales Amount"].apply(customer_segment)

    top_customers = (
        df.groupby(
            ["Customer ID", "Customer"]
        )["Sales Amount"]
        .sum()
        .reset_index()
        .sort_values(
            "Sales Amount",
            ascending=False
        )
    )

    top_customers = top_customers[
        (top_customers["Customer ID"] != "[Not Applicable]") &
        (top_customers["Customer"] != "[Not Applicable]")
    ]

    top_customers = top_customers.head(10)

    top_country = country_sales.iloc[0]["Country"]
    top_country_sales = country_sales.iloc[0]["Sales Amount"]
    top_category = category_sales.iloc[0]["Category"]
    vip_count = (segment["Segment"]=="VIP").sum()

    executive_insight = f"""
    <ul>
    <li><b>{top_country}</b> contributes the largest share of company revenue (${top_country_sales:,.0f}).</li>
    <li><b>{top_category}</b> remains the highest-performing product category.</li>
    <li><b>{vip_count:,}</b> customers are classified as VIP and represent the most valuable customer segment.</li>
    <li>CRM strategy should prioritize customer retention and personalized marketing campaigns.</li>
    </ul>
    """

    recommendations=[]
    if vip_count > total_customers*0.2:
        recommendations.append("Expand loyalty programs for VIP customers.")
    recommendations.append(f"Prioritize marketing in {top_country}.")
    recommendations.append(f"Increase focus on {top_category} category.")
    recommendation_html="".join([f"<li>{r}</li>" for r in recommendations])

    top_customers = (
    df.groupby(
        ["Customer ID", "Customer"]
    )["Sales Amount"]
    .sum()
    .reset_index()
    .sort_values(
        "Sales Amount",
        ascending=False
    )
    .head(10)
    )

    fig_customer = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Customer ID",
                        "Customer Name",
                        "Revenue ($)"
                    ],
                    fill_color="#1e293b",
                    font=dict(
                        color="white",
                        size=12
                    ),
                    align="left"
                ),
                cells=dict(
                    values=[
                        top_customers["Customer ID"],
                        top_customers["Customer"],
                        top_customers["Sales Amount"]
                        .map(lambda x: f"${x:,.0f}")
                    ],
                    align="left",
                    height=30
                )
            )
        ]
    )

    fig_customer.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0
        ),
        height=350
    )
    
    top_country = country_sales.iloc[0]["Country"]

    segment_chart = (
            segment["Segment"]
            .value_counts()
            .reset_index(name="Count")
        )

    # ปรับสีกราฟให้เข้าชุด คลีน สบายตา สไตล์โมเดิร์นแดชบอร์ด
    fig1 = px.bar(
        country_sales,
        x="Country",
        y="Sales Amount",
        title="Top Revenue Countries",
        template="plotly_white",
        color_discrete_sequence=["#3b82f6"]
    )

    fig1.update_layout(
        xaxis_tickangle=-30
    )

    fig2 = px.bar(
        category_sales,
        x="Category",
        y="Sales Amount",
        title="Top Product Categories",
        template="plotly_white",
        color_discrete_sequence=["#10b981"]
    )

    fig2.update_layout(
        xaxis_tickangle=-25
    )

    fig3 = px.line(
        yearly_sales,
        x="Fiscal Year",
        y="Sales Amount",
        markers=True,
        title="Revenue Trend over Fiscal Years",
        template="plotly_white",
        color_discrete_sequence=["#6366f1"]
    )

    fig3.update_layout(
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True))
    )

    fig3.update_traces(
        hovertemplate="Revenue: $%{y:,.0f}<extra></extra>"
    )

    fig4 = px.pie(
        segment_chart,
        names="Segment",
        values="Count",
        title="CRM Customer Segments Distribution",
        hole=0.4,
        color_discrete_sequence=[
            "#2563eb",
            "#7c3aed",
            "#10b981",
            "#f59e0b"
        ]
    )

    fig5 = px.bar(
        feature_importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Country Impact on Purchase Probability",
        template="plotly_white",
        color_discrete_sequence=["#8b5cf6"]
    )

    fig5.update_layout(
        yaxis=dict(categoryorder="total ascending")
    )

    # สร้าง Block ผลลัพธ์เริ่มต้นและกำหนดตัวแปรให้สัมพันธ์กับฟอร์ม
    buy_result_html = ""
    sales_result_html = ""
    
    selected_country_buy = unique_countries[0] if unique_countries else ""
    selected_country_sales = unique_countries[0] if unique_countries else ""
    selected_category = unique_categories[0] if unique_categories else ""
    input_price = "150"
    input_qty = "1"

    if request.method == "POST":
        action = request.form.get("action")

        if action == "buy":
            selected_country_buy = request.form.get("country")
            buy_prediction = predict_buy(clf, clf_cols, selected_country_buy)
            
            badge_class = "badge-buy" if buy_prediction == "BUY" else "badge-not"
            buy_result_html = f"""
            <div style="margin-top: 15px; text-align: center;">
                <span class="metric-title">Prediction Result:</span><br>
                <span class="badge {badge_class}">{buy_prediction}</span>
            </div>
            """

        if action == "sales":
            selected_country_sales = request.form.get("country")
            selected_category = request.form.get("category")
            input_price = request.form.get("price")
            input_qty = request.form.get("qty")

            sales_prediction = round(
                predict_sales(
                    reg,
                    reg_cols,
                    selected_country_sales,
                    selected_category,
                    float(input_price),
                    int(input_qty)
                ),
                2
            )

            if sales_prediction > 5000:
                recommendation = "💎 Premium Campaign Recommended (High Value Potential)"
            else:
                recommendation = "🏷️ Discount/Volume Campaign Recommended (Standard Potential)"
                
            sales_result_html = f"""
            <div style="margin-top: 15px; display: flex; flex-direction: column; align-items: center;">
                <span class="metric-title">Predicted Sales Amount:</span>
                <span class="metric" style="font-size:28px; color:#0ea5e9; margin-bottom:10px;">${sales_prediction:,.2f}</span>
                <div class="badge-recom">{recommendation}</div>
            </div>
            """
    
    # เจนตารางตัวเลือก Dropdowns สัมพันธ์กับค่าที่เคยกดเลือก
    country_options_buy = "".join([f'<option value="{c}" {"selected" if c==selected_country_buy else ""}>{c}</option>' for c in unique_countries])
    country_options_sales = "".join([f'<option value="{c}" {"selected" if c==selected_country_sales else ""}>{c}</option>' for c in unique_countries])
    category_options = "".join([f'<option value="{c}" {"selected" if c==selected_category else ""}>{c}</option>' for c in unique_categories])

    # ส่งออกหน้าโครงสร้าง HTML โดยใช้ f-string แสดงตัวแปรแบบเรียลไทม์
    html = f"""
    <html>
    <head>
    <title>AdventureWorks CRM Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background:#f8fafc; margin:0; color:#1e293b; }}
        header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color:white; text-align:center; padding:40px 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
        header h1 {{ margin: 0; font-size: 2.4rem; font-weight: 700; letter-spacing: -0.04em; }}
        header p {{ margin: 8px 0 0 0; opacity: 0.75; font-size: 1.05rem; font-weight: 400; }}
        .container {{ max-width: 1440px; margin: auto; padding: 25px 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin-bottom: 25px; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(550px, 1fr)); gap: 20px; margin-bottom: 25px; }}
        @media (max-width: 768px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
        .card {{ background: white; padding: 24px; border-radius: 14px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.02); border: 1px solid #e2e8f0; }}
        .metric-title {{ font-size: 0.85rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
        .metric {{ font-size: 32px; font-weight: 700; color: #0f172a; }}
        .form-group {{ margin-bottom: 14px; display: flex; flex-direction: column; }}
        label {{ font-weight: 600; font-size: 0.85rem; color: #475569; margin-bottom: 6px; }}
        select, input {{ padding: 11px 14px; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 0.95rem; background-color: #f8fafc; color:#334155; outline: none; transition: all 0.2s; }}
        select:focus, input:focus {{ border-color: #3b82f6; background-color: white; box-shadow: 0 0 0 3px rgba(59,130,246,0.1); }}
        button {{ padding: 12px; background: #3b82f6; color: white; border: none; border-radius: 8px; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: background 0.2s; margin-top: 5px; }}
        button:hover {{ background: #2563eb; }}
        .badge {{ display: inline-block; padding: 8px 16px; border-radius: 30px; font-weight: 700; font-size: 0.95rem; margin-top: 8px; letter-spacing: 0.05em; }}
        .badge-buy {{ background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }}
        .badge-not {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }}
        .badge-recom {{ background: #f0f9ff; color: #0369a1; border: 1px dashed #bae6fd; font-size: 0.92rem; padding: 12px; border-radius: 8px; width: 100%; text-align: center; font-weight: 600; margin-top:5px; }}
        .eval-item {{ display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding: 12px 0; font-size: 0.95rem; }}
        .eval-item:last-child {{ border: none; }}
    </style>
    </head>
    <body>
    <header>
        <h1>AdventureWorks CRM Analytics Dashboard</h1>
        <p>CRM Analytics & Predictive Intelligence Platform • Powered by Random Forest</p>
    </header>
    <div class="container">
        <div class="grid">
            <div class="card">
                <div class="metric-title">Total Sales Revenue</div>
                <div class="metric">${total_sales:,.0f}</div>
            </div>
            <div class="card">
                <div class="metric-title">Active Customers</div>
                <div class="metric">{total_customers:,}</div>
            </div>
            <div class="card">
                <div class="metric-title">Product Portfolio</div>
                <div class="metric">{total_products:,}</div>
            </div>
            <div class="card">
                <div class="metric-title">ML Engine</div>
                <div class="metric" style="color: #10b981;">Random Forest</div>
            </div>
        </div>
        <div class="card"><h2>📈 Executive Insights</h2>{executive_insight}</div>
        <div class="grid">
            <div class="card"><div class="metric-title">Customer Lifetime Value</div><div class="metric">${clv:,.0f}</div></div>
            <div class="card"><h3>🤖 AI Recommendations</h3><ul>{recommendation_html}</ul></div>
        </div>
        <div class="card">
            <h3>🏆 Top 10 Customers</h3>
            {fig_customer.to_html(
                full_html=False,
                config={"displayModeBar": False}
            )}
        </div>
        <div class="chart-grid">
            <div class="card">{fig1.to_html(full_html=False, config={"displayModeBar": False})}</div>
            <div class="card">{fig2.to_html(full_html=False, config={"displayModeBar": False})}</div>
        </div>
        <div class="chart-grid">
            <div class="card">
                {fig3.to_html(full_html=False,
                config={"displayModeBar": False})}
            </div>
            <div class="card">
                {fig4.to_html(full_html=False,
                config={"displayModeBar": False})}
            </div>
        </div>
        <div class="card">
            {fig5.to_html(
                full_html=False,
                config={"displayModeBar": False}
            )}
        </div>
        <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap:25px;">
            <div class="card" style="display:flex; flex-direction:column; justify-content:space-between;">
                <div>
                    <h2 style="margin-top:0; color:#0f172a; font-size:1.3rem;">🎯 Target Prospect Classification</h2>
                    <p style="color:#64748b; font-size:0.85rem; margin-bottom:20px;">Predict whether a regional customer base has high-purchasing potential.</p>
                    <form method="POST">
                        <div class="form-group">
                            <label>Target Region / Country</label>
                            <select name="country">
                                {country_options_buy}
                            </select>
                        </div>
                        <button type="submit" name="action" value="buy" style="width:100%;">Execute Classifier Model</button>
                    </form>
                </div>
                {buy_result_html}
            </div>
            <div class="card">
                <h2 style="margin-top:0; color:#0f172a; font-size:1.3rem;">📊 Revenue & Demand Forecasting</h2>
                <p style="color:#64748b; font-size:0.85rem; margin-bottom:20px;">Estimate expected sales metrics to optimize corporate resource allocation.</p>
                <form method="POST">
                    <div class="grid" style="grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom:0; padding:0;">
                        <div class="form-group">
                            <label>Country</label>
                            <select name="country">
                                {country_options_sales}
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Product Category</label>
                            <select name="category">
                                {category_options}
                            </select>
                        </div>
                    </div>
                    <div class="grid" style="grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom:0; padding:0;">
                        <div class="form-group">
                            <label>List Price ($)</label>
                            <input type="number" name="price" value="{input_price}" step="0.01">
                        </div>
                        <div class="form-group">
                            <label>Order Quantity</label>
                            <input type="number" name="qty" value="{input_qty}">
                        </div>
                    </div>
                    <button type="submit" name="action" value="sales" style="width:100%; background:#4f46e5;">Forecast Revenue</button>
                </form>
                {sales_result_html}
            </div>
        </div>
        <div class="card" style="background: #f8fafc; margin-top:10px; border: 1px dashed #cbd5e1;">
            <h2 style="margin-top:0; color:#0f172a; font-size:1.15rem;">📝 Machine Learning Evaluation & Business Value Summary</h2>
            <div class="grid" style="grid-template-columns: 1fr 1.5fr; gap: 30px; margin-bottom:0;">
                <div>
                    <div class="eval-item">
                        <strong>Classification Accuracy:</strong>
                        <span style="color:#10b981; font-weight:700;">{acc:.1%}</span>
                    </div>
                    <div class="eval-item">
                        <strong>Sales Forecast MAE:</strong>
                        <span style="color:#f59e0b; font-weight:700;">{mae:,.2f}</span>
                    </div>
                    <div class="eval-item">
                        <strong>Sales Forecast RMSE:</strong>
                        <span style="color:#ef4444; font-weight:700;">{rmse:,.2f}</span>
                    </div>
                </div>
                <div style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
                    <strong>💡 CRM Strategic Core Insights:</strong><br>
                    • <strong>Automated Campaigning:</strong> The model automatically segments and targets high-potential customer bases within individual countries, significantly reducing generalized marketing waste and maximizing ROI.<br>
                    • <strong>Risk Management & Operations:</strong> Statistical deviation metrics (MAE/RMSE) serve as data-driven benchmarks for safety stock optimization, preventing overproduction and inventory holding costs in high-volatility markets.
                </div>
            </div>
        </div>
    </div>
    </body>
    </html>
    """

    return html


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860,
        debug=False
    )
