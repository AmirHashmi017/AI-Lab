import streamlit as st
import joblib
import pandas as pd
import requests
import json

st.set_page_config(page_title="Smart Loan Default Predictor + AI Insights", page_icon="")

@st.cache_resource
def load_model():
    return joblib.load("loan_default_model.pkl")

@st.cache_resource
def load_feature_info():
    return joblib.load("feature_info.pkl")

model = load_model()
feature_info = load_feature_info()

selected_columns = feature_info['selected_columns']

categorical_options = {
    'Gender': ['Male', 'Female', 'Sex Not Available', 'Joint'] if 'Gender' in feature_info['categorical_features'] else ['Sex Not Available'],
    'Credit_Worthiness': ['good', 'poor'] if 'Credit_Worthiness' in feature_info['categorical_features'] else ['good'],
    'business_or_commercial': ['b/c', 'nob/c'] if 'business_or_commercial' in feature_info['categorical_features'] else ['nob/c']
}


default_input = {
    'loan_amount': 600000,
    'rate_of_interest': 12.0,
    'term': 360,
    'property_value': 100000,
    'income': 300,
    'Gender': 'Male',
    'Credit_Worthiness': 'poor',
    'business_or_commercial': 'b/c',
    'age': 22,
    'LTV': 600.0,  
    'DTI': 555.6   
}

with st.sidebar:
    st.header(" AI Assistant Settings")
    api_key = "AIzaSyBExQlsdTcRyxDq1fmxioEBF1wKq4-WPm8"
    st.subheader("Analysis Options")
    include_risk_analysis = st.checkbox("Include Risk Analysis", value=True)
    include_mitigation_advice = st.checkbox("Include Mitigation Strategies", value=True)
    include_feature_impact = st.checkbox("Explain Feature Impact", value=True)
    
    response_style = st.selectbox(
        "Response Style",
        ["Professional", "Casual", "Technical", "Beginner-Friendly"],
        index=0
    )

st.title(" Smart Loan Default Predictor + AI Insights")
st.markdown("Enter loan and applicant details to predict default risk with AI-powered insights.")

# Input form
st.subheader("Loan and Applicant Details")
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        loan_amount = st.number_input("Loan Amount ($)", min_value=10000, max_value=1000000, value=default_input['loan_amount'], step=1000)
        rate_of_interest = st.number_input("Interest Rate (%)", min_value=0.0, max_value=20.0, value=default_input['rate_of_interest'], step=0.1)
        term = st.selectbox("Loan Term (months)", [180, 240, 360], index=2)
        property_value = st.number_input("Property Value ($)", min_value=10000, max_value=2000000, value=default_input['property_value'], step=1000)
        income = st.number_input("Monthly Income ($)", min_value=100, max_value=20000, value=default_input['income'], step=100)
    
    with col2:
        gender = st.selectbox("Gender", categorical_options['Gender'], index=categorical_options['Gender'].index(default_input['Gender']))
        credit_worthiness = st.selectbox("Credit Worthiness", categorical_options['Credit_Worthiness'], index=categorical_options['Credit_Worthiness'].index(default_input['Credit_Worthiness']))
        business_or_commercial = st.selectbox("Business/Commercial Loan", categorical_options['business_or_commercial'], index=categorical_options['business_or_commercial'].index(default_input['business_or_commercial']))
        age = st.number_input("Age", min_value=18, max_value=80, value=default_input['age'], step=1)
    
    submit_button = st.form_submit_button(" Predict & Analyze", type="primary")

def get_gemini_insights(loan_features, predicted_proba, user_inputs):
    """Get detailed insights from Gemini AI about the loan default prediction"""
    
    if not api_key.strip():
        return "Please provide a valid Gemini API key in the sidebar to get AI insights."
    
    style_prompts = {
        "Professional": "Provide a professional loan risk analysis",
        "Casual": "Explain in a friendly, conversational tone",
        "Technical": "Give a detailed technical analysis with specific metrics",
        "Beginner-Friendly": "Explain in simple terms for someone new to loans"
    }
    
    prompt_parts = [
        f"{style_prompts[response_style]} for this loan default prediction:",
        f"Default Probability: {predicted_proba[1]:.2%} (No Default: {predicted_proba[0]:.2%})",
        f"Loan Features: {dict(loan_features.iloc[0])}",
        f"User Inputs: {user_inputs}"
    ]
    
    if include_risk_analysis:
        prompt_parts.append("Include risk analysis and probability justification.")
    
    if include_mitigation_advice:
        prompt_parts.append("Provide strategies to mitigate default risk.")
    
    if include_feature_impact:
        prompt_parts.append("Explain how each feature impacts the default risk.")
    
    prompt_parts.extend([
        "Keep the response comprehensive but concise (300-500 words).",
        "Focus on actionable insights and practical advice."
    ])
    
    prompt = " ".join(prompt_parts)
    
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in response_json:
            return f"❌ Gemini API Error: {response_json['error'].get('message', 'Unknown error occurred.')}"
        else:
            return "❌ Unexpected response format from Gemini API."
            
    except requests.exceptions.Timeout:
        return "❌ Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"❌ HTTP Error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {str(e)}"
    except Exception as e:
        return f"❌ Error getting AI insights: {str(e)}"

if submit_button:
    input_data = {
        'loan_amount': loan_amount,
        'rate_of_interest': rate_of_interest,
        'term': term,
        'property_value': property_value,
        'income': income,
        'Gender': gender,
        'Credit_Worthiness': credit_worthiness,
        'business_or_commercial': business_or_commercial,
        'age': age,
        'LTV': (loan_amount / property_value * 100) if property_value > 0 else 0,
        'DTI': (loan_amount / term / income * 100) if income > 0 else 0
    }
    input_df = pd.DataFrame([input_data])[selected_columns]
    
    with st.expander("📋 Input Loan Details"):
        st.dataframe(input_df)
    
    try:
        proba = model.predict_proba(input_df)[0]
        prediction = 1 if proba[1] > 0.25 else 0
        
        col1, col2, col3 = st.columns(3)
        with col2:
            risk_level = "High" if proba[1] > 0.5 else "Moderate" if proba[1] > 0.3 else "Low"
            result_text = '<span style="color: red;">Default</span>' if prediction == 1 else '<span style="color: white;">No Default</span>'
            st.markdown(f"""
            <div style="text-align: center; padding: 8px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 10px auto; width: 100%; max-width: 400px; box-sizing: border-box;">
                <h4 style="margin: 0; color: white; font-size: 1.1rem;"> Default Risk</h4>
                <h4 style="margin: 4px 0; font-size: 1rem;">{result_text}</h4>
                <h4 style="margin: 4px 0; color: white; font-size: 1rem;">Risk Level: {risk_level}</h4>
                <h4 style="margin: 4px 0; color: white; font-size: 1rem;">Default Probability: {proba[1]:.2%}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner(" Getting AI insights..."):
            user_inputs = (f"Loan Amount: ${loan_amount:,}, Interest Rate: {rate_of_interest}%, "
                          f"Term: {term} months, Property Value: ${property_value:,}, "
                          f"Income: ${income:,}/month, Gender: {gender}, "
                          f"Credit Worthiness: {credit_worthiness}, "
                          f"Business/Commercial: {business_or_commercial}, Age: {age}, "
                          f"LTV: {input_data['LTV']:.1f}%, DTI: {input_data['DTI']:.1f}%")
            insights = get_gemini_insights(input_df, proba, user_inputs)
        
        st.subheader(" AI-Powered Risk Analysis")
        st.markdown(insights)
        
        st.subheader(" Additional Insights")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Loan-to-Value Ratio", f"{input_data['LTV']:.1f}%")
        
        with col2:
            st.metric("Debt-to-Income Ratio", f"{input_data['DTI']:.1f}%")
        
        with col3:
            st.metric("Loan Term", f"{term // 12} years")
        
        with col4:
            st.metric("Age", f"{age} years")
            
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")