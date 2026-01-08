import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    model = pickle.load(open("model/xgb_model.pkl", "rb"))
    features = pickle.load(open("model/feature_columns.pkl", "rb"))
    return model, features

model, feature_columns = load_model()

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction App")
st.markdown("""
Predict which customers are likely to churn using **machine learning**.  
This app is powered by XGBoost and feature engineering techniques.
""")


st.subheader("Customer Information (Top 6 Key Features)")

# 1. Contract
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# 2. Tenure
tenure = st.slider("Tenure (months)", 0, 72, 12)

# 3. Monthly Charges
monthly_charges = st.number_input("Monthly Charges ($)", 18.25, 150.0, 70.0)

# 4. Internet Service
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# 5. Online Security
online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])

# 6. Tech Support
tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])

input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "Contract_One year": 1 if contract=="One year" else 0,
    "Contract_Two year": 1 if contract=="Two year" else 0,
    "InternetService_Fiber optic": 1 if internet=="Fiber optic" else 0,
    "InternetService_No": 1 if internet=="No" else 0,
    "OnlineSecurity_Yes": 1 if online_security=="Yes" else 0,
    "TechSupport_Yes": 1 if tech_support=="Yes" else 0
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)



if st.button("Predict Churn"):

    prob = model.predict_proba(input_df)[0][1]

    # --- Probability Display ---
    st.subheader("📊 Churn Probability")
    st.progress(int(prob * 100))
    st.write(f"**Churn Probability:** {prob:.2%}")

    # --- Risk Label ---
    if prob >= 0.7:
        st.error("⚠️ High Risk Customer")
        risk = "High"
    elif prob >= 0.4:
        st.warning("⚠️ Medium Risk Customer")
        risk = "Medium"
    else:
        st.success("✅ Low Risk Customer")
        risk = "Low"

    # --- Retention Recommendations ---
    st.subheader("🛠 Recommended Retention Actions")

    if risk == "High":
        st.markdown("""
        - Offer personalized discount or loyalty bonus  
        - Assign retention agent  
        - Incentivize long-term contract  
        - Address service issues immediately  
        """)
    elif risk == "Medium":
        st.markdown("""
        - Send targeted promotions  
        - Encourage contract upgrade  
        - Highlight service benefits  
        """)
    else:
        st.markdown("""
        - Maintain engagement  
        - Offer loyalty rewards  
        - Monitor periodically  
        """)
        


    # --- Download Report ---
    business_report = {
    "Contract Type": contract,
    "Tenure (Months)": tenure,
    "Monthly Charges ($)": monthly_charges,
    "Internet Service": internet,
    "Online Security": online_security,
    "Tech Support": tech_support,
    "Churn Probability": round(prob, 3),
    "Risk Level": risk,
    "Business Insight 1": "Offer contract upgrade" if contract=="Month-to-month" else "",
    "Business Insight 2": "Focus on onboarding" if tenure<12 else "",
    "Business Insight 3": "Provide premium support" if tech_support!="Yes" else ""
     }

    report_df = pd.DataFrame([business_report])




    st.download_button(
    "📥 Download Business Churn Report (CSV)",
    report_df.to_csv(index=False),
    "business_churn_report.csv",
    "text/csv"
    )


st.subheader("💡 Feature-Based Insights")

# Contract
if contract == "Month-to-month":
    st.write("- Customer is on a short-term contract — higher churn risk. Consider incentives to upgrade.")

# Tenure
if tenure < 12:
    st.write("- New customer (less than 1 year) — at higher risk, focus on onboarding.")

# Monthly Charges
if monthly_charges > 80:
    st.write("- High monthly charges — may churn if not seeing value, offer bundled services or discounts.")

# Internet
if internet == "Fiber optic":
    st.write("- Customer has premium internet — ensure top service quality to reduce churn.")

# Online Security
if online_security != "Yes":
    st.write("- Customer lacks Online Security — upsell or educate about security benefits.")

# Tech Support
if tech_support != "Yes":
    st.write("- Customer has limited Tech Support — proactive support can reduce churn.")
    



st.subheader("🔍 Top Factors Influencing Churn")

importance = model.feature_importances_

fi_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False).head(10)

st.bar_chart(fi_df.set_index("Feature"))


st.write("This app predicts whether a customer is likely to churn.")

st.markdown("""
### 📌 Business Insights
- Contract type and tenure are the strongest churn drivers
- Month-to-month customers are at the highest risk
- Higher monthly charges increase churn probability
- Long-term contracts significantly reduce churn risk
""")


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>⚡ This app is powered by <b>XGBoost</b> and feature engineering</p>",
    unsafe_allow_html=True
)

