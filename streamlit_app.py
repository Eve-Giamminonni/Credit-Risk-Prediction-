import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load Model
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
THRESHOLD = 0.40  # Decision threshold



# PAGE 1 — HOME / LANDING PAGE

def page_home():
    
    st.markdown(
        """
        <h1 style="
            text-align:center;
            font-size:64px;
            font-weight:700;
            color:white;
            margin-bottom:10px;
        ">
            Credit Risk Prediction Platform
        </h1>

        <p style="
            text-align:center;
            font-size:20px;
            color:#A5A5A5;
            margin-top:0;
        ">
            Financial Risk Scoring • Loan Assessment • Decision Support
        </p>
        """,
        unsafe_allow_html=True
    )


    
    st.markdown(
        """
        <div class="description-card">
            This application predicts the credit risk of American companies 
            using financial ratios and the loan amount they request.<br><br>
            It is designed as a decision-support tool for banks evaluating whether 
            a company is likely to repay a loan.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    
    st.image("risk_image.png", use_container_width=False)


    
    st.markdown(
        """
        <p style="
            text-align:center;
            margin-top:80px;
            color:#777;
            font-size:15px;
            font-style:italic;
        ">
            By Eve Giamminonni
        </p>
        """,
        unsafe_allow_html=True
    )

    



# PAGE 2 — PREDICTION TOOL

def page_prediction():
    st.markdown("""
    <h1 style='text-align:center; font-size:48px; font-weight:700; color:white;'>
        Loan Risk Prediction Tool
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align:center; font-size:18px; color:#A5A5A5; margin-top:-10px;'>
        Provide the company’s key financial ratios.<br>
        This tool will evaluate the probability of repayment<br>
        and indicate whether the loan should be approved.
    </p>
    """, unsafe_allow_html=True)

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        EBITDA = st.number_input("EBITDA", value=0.0)
        Inventory = st.number_input("Inventory", value=0.0)
        Net_Income = st.number_input("Net Income", value=0.0)
        Total_Receivables = st.number_input("Total Receivables", value=0.0)
        Total_Assets = st.number_input("Total Assets", value=0.0)

    with col2:
        Market_Value = st.number_input("Market Value", value=0.0)
        Gross_Profit = st.number_input("Gross Profit", value=0.0)
        Current_Liabilities = st.number_input("Current Liabilities", value=0.0)
        Retained_Earnings = st.number_input("Retained Earnings", value=0.0)
        Total_Liabilities = st.number_input("Total Liabilities", value=0.0)

    st.write("---")

    loan_amount = st.number_input("Loan Amount Requested", value=0.0)
    loan_ratio = 0 if Total_Assets == 0 else loan_amount / Total_Assets

    st.metric("Loan Ratio (computed)", f"{loan_ratio:.4f}")
    st.write("---")

    if st.button("Predict Risk"):
        all_features = [
            "Current_Assets", "Cost_of_Goods_Sold", "Depreciation_Amortization",
            "EBITDA", "Inventory", "Net_Income", "Total_Receivables", "Market_Value",
            "Net_Sales", "Total_Assets", "Long_Term_Debt", "EBIT", "Gross_Profit",
            "Current_Liabilities", "Retained_Earnings", "Total_Revenue",
            "Total_Liabilities", "Operating_Expenses", "loan_amount", "loan_ratio"
        ]

        data_dict = {f: 0 for f in all_features}
        data_dict.update({
            "EBITDA": EBITDA,
            "Inventory": Inventory,
            "Net_Income": Net_Income,
            "Total_Receivables": Total_Receivables,
            "Market_Value": Market_Value,
            "Gross_Profit": Gross_Profit,
            "Current_Liabilities": Current_Liabilities,
            "Retained_Earnings": Retained_Earnings,
            "Total_Assets": Total_Assets,
            "Total_Liabilities": Total_Liabilities,
            "loan_amount": loan_amount,
            "loan_ratio": loan_ratio
        })

        input_data = pd.DataFrame([[data_dict[f] for f in all_features]], columns=all_features)

        proba = model.predict_proba(input_data)[0]
        prob_alive = proba[0]
        prob_failed = proba[1]

        st.subheader("Prediction Results")
        st.write(f"Probability of Repayment: `{prob_alive:.3f}`")
        st.write(f"Probability of Default: `{prob_failed:.3f}`")

        st.write("---")

        prediction = 1 if prob_failed >= THRESHOLD else 0

        if prediction == 1:
            st.markdown("""
                <div class="error-card">
                    <h3>Loan NOT Recommended</h3>
                    High probability of default. The company presents a financial risk.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="success-card">
                    <h3>Loan Approved</h3>
                    Low risk — The company is financially healthy enough to receive the loan.
                </div>
            """, unsafe_allow_html=True)



# PAGE 3 — ABOUT THE PROJECT

def page_about():
    st.markdown("""
        <h1 style="text-align:center; font-size:48px; font-weight:700; color:white;">
            About This Project
        </h1>
    """, unsafe_allow_html=True)

    
    with st.expander("Dataset"):
        st.markdown("""
        The model is trained on the **American Company Bankruptcy Prediction Dataset** (Kaggle), which includes:
        - ~78,000 U.S. companies  
        - 18 key financial ratios  
        - A target variable indicating **bankrupt vs. healthy**

        These ratios capture profitability, leverage, liquidity, and operational performance — all commonly used by banks during credit risk analysis.
        """)

    
    with st.expander("Model"):
        st.markdown("""
        The final model is an **optimized CatBoost Classifier**, chosen for its strong performance on financial tabular data and its robustness to class imbalance.

        **Key performance:**  
        - F1-score (bankrupt): ~0.55  
        - F1-score (healthy): ~0.97  

        The model estimates the probability that a company defaults or remains solvent based on its financial structure.
        """)

    
    with st.expander("Default Probability"):
        st.markdown("""
        For each company, the model outputs:
        - **Probability of Repayment**  
        - **Probability of Default**

        A custom threshold adapted to banking risk tolerance is then used to produce a final recommendation:  
        **approve** or **reject** the loan request.
        """)

    
    with st.expander("Engineered Features"):
        st.markdown("""
        To simulate real banking practices, two additional variables were engineered:

        **• loan_amount**  
        A simulated representation of the amount the company wishes to borrow.

        **• loan_ratio = loan_amount ÷ Total_Assets**  
        This ratio evaluates the size of the loan relative to the company's asset base — higher ratios typically indicate higher financial risk.

        These additions help the model adjust risk based on both the company’s financial health and the scale of the loan, creating a more realistic credit-risk scoring system.
        """)

    
    st.write("---")

    st.subheader("Feature Importance")

    try:
        importances = model.get_feature_importance()
        feature_names = model.feature_names_

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(feature_names, importances, color="#4BA3C3")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        ax.set_title("CatBoost Feature Importance")
        plt.tight_layout()

        st.pyplot(fig)
    except:
        st.warning("Feature importance unavailable.")



# SIDEBAR NAVIGATION

pages = {
    "Home": page_home,
    "Loan Risk Prediction": page_prediction,
    "About This Project": page_about
}

st.sidebar.markdown("""
<div class="sidebar-title-fintech">
    Dashboard
    <div class="sidebar-underline"></div>
</div>
""", unsafe_allow_html=True)

# Navigation radio (no label)
choice = st.sidebar.radio("", list(pages.keys()))

pages[choice]()