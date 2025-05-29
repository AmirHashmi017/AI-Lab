import streamlit as st
import joblib
import pandas as pd
import re
import requests
import json

st.set_page_config(page_title="Smart House Price Predictor + AI Insights", page_icon="🏡")
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()





selected_columns = [
    'BedroomAbvGr', 'FullBath', 'KitchenAbvGr', 'PoolArea', 'LotArea',
    'YearBuilt', 'GarageCars', 'OverallQual', 'Neighborhood'
]


default_input = {
    "BedroomAbvGr": 3,
    "FullBath": 2,
    "KitchenAbvGr": 1,
    "LotArea": 7000,
    "PoolArea": 0,
    "YearBuilt": 2000,
    "GarageCars": 1,
    "OverallQual": 6,
    "Neighborhood": "CollgCr"
}


with st.sidebar:
    st.header("⚙️ AI Assistant Settings")
    api_key = "AIzaSyBExQlsdTcRyxDq1fmxioEBF1wKq4-WPm8"
    
    st.subheader("Analysis Options")
    include_market_analysis = st.checkbox("Include Market Analysis", value=True)
    include_investment_advice = st.checkbox("Include Investment Insights", value=True)
    include_feature_impact = st.checkbox("Explain Feature Impact", value=True)
    
    response_style = st.selectbox(
        "Response Style",
        ["Professional", "Casual", "Technical", "Beginner-Friendly"],
        index=0
    )

st.title("🏡 Smart House Price Predictor + AI Insights")
st.markdown("Describe your dream house and get AI-powered price predictions with detailed market analysis")

user_text = st.text_area("Chat with the model", placeholder="E.g. I want a house with 4 bedrooms, 2 baths, 1 kitchen, pool, and a garage in CollgCr neighborhood")

def parse_input(text):
    inp = default_input.copy()
    text = text.lower()

    patterns = {
        "BedroomAbvGr": r'(\d+)\s*bed(room)?s?',
        "FullBath": r'(\d+)\s*(bath|bathroom)s?',
        "KitchenAbvGr": r'(\d+)\s*kitchen',
        "PoolArea": r'pool',
        "LotArea": r'(\d{4,6})\s*(sqft|square feet|lot)',
        "YearBuilt": r'built\s+in\s+(\d{4})',
        "GarageCars": r'(\d+)\s*car\s*garage',
        "OverallQual": r'(?:quality|overall).{0,10}?(\d+)',
        "Neighborhood": r'in\s+(\w+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            if key == "PoolArea":
                inp[key] = 1
            elif key == "Neighborhood":
                inp[key] = match.group(1).capitalize()
            else:
                try:
                    inp[key] = int(match.group(1))
                except:
                    pass

    df = pd.DataFrame([inp])
    return df[selected_columns]

def get_gemini_insights(house_features, predicted_price, user_preferences):
    """Get detailed insights from Gemini AI about the house price prediction"""
    
    if not api_key.strip():
        return "Please provide a valid Gemini API key in the sidebar to get AI insights."
    
    style_prompts = {
        "Professional": "Provide a professional real estate analysis",
        "Casual": "Explain in a friendly, conversational tone",
        "Technical": "Give a detailed technical analysis with specific metrics",
        "Beginner-Friendly": "Explain in simple terms for someone new to real estate"
    }
    
    prompt_parts = [
        f"{style_prompts[response_style]} for this house prediction:",
        f"Predicted Price: ${predicted_price:,.2f}",
        f"House Features: {dict(house_features.iloc[0])}",
        f"User Description: {user_preferences}"
    ]
    
    if include_market_analysis:
        prompt_parts.append("Include market analysis and price justification.")
    
    if include_investment_advice:
        prompt_parts.append("Provide investment insights and potential ROI considerations.")
    
    if include_feature_impact:
        prompt_parts.append("Explain how each feature impacts the house value.")
    
    prompt_parts.extend([
        "Keep the response comprehensive but concise (300-500 words).",
        "Focus on actionable insights and practical advice."
    ])
    
    prompt = " ".join(prompt_parts)
    
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
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
        response_json = response.json()
        
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        elif "error" in response_json:
            return f"❌ Gemini API Error: {response_json['error'].get('message', 'Unknown error occurred.')}"
        else:
            return "❌ Unexpected response format from Gemini API."
            
    except requests.exceptions.Timeout:
        return "❌ Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {str(e)}"
    except Exception as e:
        return f"❌ Error getting AI insights: {str(e)}"

if st.button("🔮 Predict & Analyze", type="primary"):
    if user_text.strip() == "":
        st.warning("Please describe your house to get a prediction.")
    else:
        input_df = parse_input(user_text)

        with st.expander("📋 Parsed House Features"):
            st.dataframe(input_df)

        try:
            prediction = model.predict(input_df)[0]

            col1, col2, col3 = st.columns(3)
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin: 20px 0;">
                    <h3 style="margin: 0; color: white;">💰 Estimated Price</h2>
                    <h3 style="margin: 10px 0; color: white;">${prediction:,.0f}</h1>
                    <h3 style="margin: 10px 0; color: white;">Accuracy: 83.43%</h1>
                </div>
                """, unsafe_allow_html=True)

            with st.spinner("🤖 Getting AI insights..."):
                insights = get_gemini_insights(input_df, prediction, user_text)

            st.subheader("🧠 AI-Powered Market Analysis")
            st.markdown(insights)

            st.subheader("📊 Additional Insights")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                price_per_sqft = prediction / input_df['LotArea'].iloc[0] if input_df['LotArea'].iloc[0] > 0 else 0
                st.metric("Price per Sq Ft", f"${price_per_sqft:.2f}")
            
            with col2:
                house_age = 2024 - input_df['YearBuilt'].iloc[0]
                st.metric("House Age", f"{house_age} years")
            
            with col3:
                total_rooms = input_df['BedroomAbvGr'].iloc[0] + input_df['FullBath'].iloc[0]
                st.metric("Total Rooms", f"{total_rooms}")
            
            with col4:
                has_pool = "Yes" if input_df['PoolArea'].iloc[0] > 0 else "No"
                st.metric("Pool", has_pool)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

