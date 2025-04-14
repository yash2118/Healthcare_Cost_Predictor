import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load models and artifacts
model = joblib.load('healthcare_cost_model.pkl')
scaler = joblib.load('scaler.pkl')
X_train = joblib.load('X_train.pkl')
feature_names = joblib.load('feature_names.pkl')
label_encoders = {
    col: joblib.load(f'label_encoder_{col}.pkl') 
    for col in ['sex', 'smoker', 'region', 'alcohol_consumption',
                'exercise_frequency', 'employment_status',
                'insurance_type', 'education_level']
}

# Risk thresholds and cost data
RISK_THRESHOLDS = {
    'Low': 15000,
    'Medium': 30000,
    'High': float('inf')
}

TIER_COSTS = {
    'Low': 12000,
    'Medium': 22000,
    'High': 45000
}

PROACTIVE_STRATEGIES = {
    'Low': ["Wellness program discounts", "Preventive screening reminders"],
    'Medium': ["Chronic condition monitoring", "Telehealth consultations"],
    'High': ["Personalized care teams", "Priority specialist access"]
}

# App configuration
st.set_page_config(page_title="Healthcare Cost Predictor", layout="wide")
st.title("🏥 Healthcare Cost Prediction & Insurance Optimization Tool")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["Cost Prediction", "Cost Analysis", "Smoking Cessation Impact", "Premium Optimizer"])

# Tab 1: Cost Prediction
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Patient Profile")
        bmi = st.slider('BMI', 15.0, 50.0, 25.0)
        smoker = st.selectbox('Smoker', label_encoders['smoker'].classes_)
        region = st.selectbox('Region', label_encoders['region'].classes_)
        chronic_conditions = st.slider('Chronic Conditions', 0, 10, 0)

    with col2:
        st.header("Health Indicators")
        alcohol = st.selectbox('Alcohol Consumption', label_encoders['alcohol_consumption'].classes_)
        exercise = st.selectbox('Exercise Frequency', label_encoders['exercise_frequency'].classes_)
        hospital_visits = st.slider('Hospital Visits (Last Year)', 0, 20, 0)
        mental_health = st.slider('Mental Health Score', 0, 100, 75)

    # Create input data with default values for potentially biased features
    input_dict = {
        'age': 35,  # Default middle age
        'bmi': bmi,
        'children': 0,  # Default to zero
        'smoker': label_encoders['smoker'].transform([smoker])[0],
        'region': label_encoders['region'].transform([region])[0],
        'chronic_conditions': chronic_conditions,
        'alcohol_consumption': label_encoders['alcohol_consumption'].transform([alcohol])[0],
        'exercise_frequency': label_encoders['exercise_frequency'].transform([exercise])[0],
        'hospital_visits_last_year': hospital_visits,
        'mental_health_score': mental_health,
        'income_level': 50000,  # Default middle income
        'sex': 0,  # Default neutral
        'employment_status': label_encoders['employment_status'].transform(['employed'])[0],
        'insurance_type': label_encoders['insurance_type'].transform(['private'])[0],
        'education_level': label_encoders['education_level'].transform(['bachelor'])[0]
    }
    
    input_data = pd.DataFrame([input_dict])[feature_names]
    
    numerical_cols = ['age', 'bmi', 'children', 'income_level', 
                     'hospital_visits_last_year', 'mental_health_score']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    if st.button('Predict Cost'):
        prediction = model.predict(input_data)[0]
        
        risk_tier = next(
            (tier for tier, threshold in RISK_THRESHOLDS.items() 
             if prediction <= threshold),
            'High'
        )
        
        # Create columns for results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader(f"Predicted Annual Cost: :green[${prediction:,.2f}]")
            st.subheader(f"Risk Tier: :{risk_tier.lower()}[\"{risk_tier}\"]")
            
            # Recommended premium
            margin = 0.15  # 15% profit margin
            monthly_premium = (prediction * (1 + margin)) / 12
            st.metric("Recommended Monthly Premium", f"${monthly_premium:.2f}")
            
            st.markdown("### 🛎️ Proactive Care Strategies")
            for strategy in PROACTIVE_STRATEGIES[risk_tier]:
                st.markdown(f"- {strategy}")
        
        with res_col2:
            # SHAP explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_data)

            explanation = shap.Explanation(
                values=shap_values.values[0], 
                base_values=shap_values.base_values[0], 
                data=input_data.iloc[0], 
                feature_names=feature_names
            )
            
            plt.figure()
            shap.plots.waterfall(explanation, max_display=8)
            st.pyplot(plt.gcf())

# Tab 2: Cost Analysis
with tab2:
    st.header("📊 Risk Tier Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SHAP summary plot
        st.subheader("Cost Drivers Across Patients")
        explainer = shap.TreeExplainer(model)
        shap_values_global = explainer.shap_values(X_train[:100])
        
        plt.figure()
        shap.summary_plot(shap_values_global, X_train[:100], show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
    
    with col2:
        # Risk tier cost distribution - this generates the chart in the image
        st.subheader("Cost Distribution by Risk Tier")
        
        # Create dataframe for cost distribution
        data_cost_distribution = pd.DataFrame({
            "Tier": ["Low", "Medium", "High"],
            "Avg Cost": [TIER_COSTS['Low'], TIER_COSTS['Medium'], TIER_COSTS['High']]
        })
        
        # Generate bar chart exactly as shown in the image
        plt.figure(figsize=(8, 6))
        bars = plt.bar(data_cost_distribution['Tier'], data_cost_distribution['Avg Cost'], 
                color=['green', 'orange', 'red'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${int(height):,}',
                    ha='center', va='bottom')
        
        plt.xlabel("Risk Tier")
        plt.ylabel("Average Annual Cost ($)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        st.pyplot(plt.gcf())
        
        # Population distribution
        st.subheader("Population Distribution by Risk Tier")
        plt.figure(figsize=(6, 6))
        plt.pie([65, 25, 10], labels=['Low', 'Medium', 'High'], 
                colors=['green', 'orange', 'red'], autopct='%1.1f%%')
        plt.title("Population Distribution")
        plt.tight_layout()
        
        st.pyplot(plt.gcf())

# Tab 3: Smoking Cessation Impact
with tab3:
    st.header("🚭 Smoking Cessation Impact Analysis")
    
    st.write("This analysis shows how quitting smoking affects healthcare costs and risk tier classification.")
    
    bmi = st.slider('BMI', 15.0, 50.0, 25.0, key="smoking_bmi")
    chronic_conditions = st.slider('Chronic Conditions', 0, 10, 0, key="smoking_cc")
    
    # Create two scenarios: current smoker and former smoker
    base_dict = {
        'age': 35,
        'bmi': bmi,
        'children': 0,
        'region': label_encoders['region'].transform(['northeast'])[0],
        'chronic_conditions': chronic_conditions,
        'alcohol_consumption': label_encoders['alcohol_consumption'].transform(['moderate'])[0],
        'exercise_frequency': label_encoders['exercise_frequency'].transform(['regular'])[0],
        'hospital_visits_last_year': 1,
        'mental_health_score': 75,
        'income_level': 50000,
        'sex': 0,
        'employment_status': label_encoders['employment_status'].transform(['employed'])[0],
        'insurance_type': label_encoders['insurance_type'].transform(['private'])[0],
        'education_level': label_encoders['education_level'].transform(['bachelor'])[0]
    }
    
    # Create smoking scenario
    smoking_dict = base_dict.copy()
    smoking_dict['smoker'] = label_encoders['smoker'].transform(['yes'])[0]
    smoking_data = pd.DataFrame([smoking_dict])[feature_names]
    
    # Create non-smoking scenario
    nonsmoking_dict = base_dict.copy()
    nonsmoking_dict['smoker'] = label_encoders['smoker'].transform(['no'])[0]
    nonsmoking_data = pd.DataFrame([nonsmoking_dict])[feature_names]
    
    # Scale numerical features
    smoking_data[numerical_cols] = scaler.transform(smoking_data[numerical_cols])
    nonsmoking_data[numerical_cols] = scaler.transform(nonsmoking_data[numerical_cols])
    
    # Make predictions
    smoking_cost = model.predict(smoking_data)[0]
    nonsmoking_cost = model.predict(nonsmoking_data)[0]
    
    # Determine risk tiers
    smoking_tier = next(
        (tier for tier, threshold in RISK_THRESHOLDS.items() 
         if smoking_cost <= threshold),
        'High'
    )
    
    nonsmoking_tier = next(
        (tier for tier, threshold in RISK_THRESHOLDS.items() 
         if nonsmoking_cost <= threshold),
        'High'
    )
    
    # Calculate savings
    savings = smoking_cost - nonsmoking_cost
    savings_percent = (savings / smoking_cost) * 100
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current Smoker Cost", f"${smoking_cost:,.2f}")
        st.metric("Current Risk Tier", smoking_tier, 
                  delta_color="off")
        
        st.metric("After Quitting Cost", f"${nonsmoking_cost:,.2f}", 
                  f"-${savings:,.2f} ({savings_percent:.1f}%)")
        st.metric("New Risk Tier", nonsmoking_tier,
                  "Improved" if smoking_tier != nonsmoking_tier else "Same")
        
        st.write(f"### Annual Savings: ${savings:,.2f}")
        st.write(f"### 5-Year Projected Savings: ${savings*5:,.2f}")
        
    with col2:
        # Create bar chart comparing costs
        plt.figure(figsize=(10, 6))
        bars = plt.bar(['Current Smoker', 'After Quitting'], 
                [smoking_cost, nonsmoking_cost],
                color=['#FF6B6B', '#4CAF50'])
        
        # Add threshold lines
        plt.axhline(y=RISK_THRESHOLDS['Low'], color='green', linestyle='--', 
                    label=f'Low Risk Threshold (${RISK_THRESHOLDS["Low"]:,})')
        plt.axhline(y=RISK_THRESHOLDS['Medium'], color='orange', linestyle='--',
                    label=f'Medium Risk Threshold (${RISK_THRESHOLDS["Medium"]:,})')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${int(height):,}',
                    ha='center', va='bottom')
        
        plt.xlabel('Smoking Status')
        plt.ylabel('Annual Healthcare Cost ($)')
        plt.title('Impact of Smoking Cessation on Healthcare Costs')
        plt.legend()
        plt.tight_layout()
        
        st.pyplot(plt.gcf())

# Tab 4: Premium Optimizer
with tab4:
    st.header("💲 Premium Optimization & Portfolio Management")
    
    st.markdown("""
    This tool helps optimize premium pricing strategies and manage risk across your customer portfolio.
    """)
    
    # Premium calculation section
    st.subheader("Premium Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_tier = st.selectbox("Risk Tier", ["Low", "Medium", "High"])
        base_cost = TIER_COSTS[risk_tier]
        profit_margin = st.slider("Profit Margin (%)", 5, 30, 15)
        discount_rate = st.slider("Wellness Discount (%)", 0, 20, 5)
    
    with col2:
        # Calculate premiums
        annual_premium = base_cost * (1 + profit_margin/100)
        monthly_premium = annual_premium / 12
        discounted_monthly = monthly_premium * (1 - discount_rate/100)
        
        # Display premium options
        st.metric("Standard Monthly Premium", f"${monthly_premium:.2f}")
        st.metric("With Wellness Discount", f"${discounted_monthly:.2f}", 
                  f"-${monthly_premium - discounted_monthly:.2f}")
        
        expected_profit = base_cost * (profit_margin/100)
        st.metric("Expected Annual Profit", f"${expected_profit:.2f}")
    
    # Portfolio optimization
    st.subheader("Portfolio Risk Management")
    
    st.markdown("Adjust your customer portfolio mix to optimize profitability while managing risk:")
    
    # Sliders for customer mix
    low_pct = st.slider("Low-Risk Customers (%)", 30, 80, 65)
    med_pct = st.slider("Medium-Risk Customers (%)", 10, 50, 25)
    high_pct = st.slider("High-Risk Customers (%)", 5, 30, 10, 
                        disabled=True, help="Automatically calculated")
    
    # Force percentages to add up to 100%
    high_pct = 100 - low_pct - med_pct
    
    # Recalculate if high percentage is negative
    if high_pct < 0:
        st.error("Invalid percentages. Please adjust Low and Medium risk groups to sum to 100% or less.")
    else:
        # Calculate portfolio metrics
        portfolio_size = 10000  # Assumption
        
        portfolio_data = pd.DataFrame({
            "Risk Tier": ["Low", "Medium", "High"],
            "Percentage": [low_pct, med_pct, high_pct],
            "Members": [portfolio_size * low_pct/100, 
                       portfolio_size * med_pct/100, 
                       portfolio_size * high_pct/100],
            "Avg Cost": [TIER_COSTS["Low"], TIER_COSTS["Medium"], TIER_COSTS["High"]]
        })
        
        portfolio_data["Total Cost"] = portfolio_data["Members"] * portfolio_data["Avg Cost"]
        
        # Calculate aggregate metrics
        total_cost = portfolio_data["Total Cost"].sum()
        avg_portfolio_cost = total_cost / portfolio_size
        
        # Display portfolio metrics
        st.dataframe(portfolio_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Member Cost", f"${avg_portfolio_cost:,.2f}")
            st.metric("Total Annual Claims", f"${total_cost:,.2f}")
        
        with col2:
            # Create portfolio mix pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(portfolio_data["Percentage"], 
                    labels=portfolio_data["Risk Tier"],
                    colors=["green", "orange", "red"],
                    autopct='%1.1f%%',
                    startangle=90)
            plt.title("Portfolio Risk Distribution")
            plt.tight_layout()
            
            st.pyplot(plt.gcf())
