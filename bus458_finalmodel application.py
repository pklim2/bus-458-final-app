import streamlit as st
import pickle
import pandas as pd

# Load the regression model
with open(r"bus458_finalmodel.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app title
st.markdown(
    "<h1 style='text-align: center; background-color: #e0f7fa; padding: 10px; color: #006064;'><b>Data Scientist Salary Predictor</b></h1>",
    unsafe_allow_html=True
)

# Header
st.header("Enter Details for Compensation Prediction")

# Numerical input fields
years_coding = st.slider("Years of Coding Experience", min_value=0, max_value=50, value=5)
years_ml = st.slider("Years of Machine Learning Experience", min_value=0, max_value=50, value=2)
money_spent = st.number_input("Money Spent on ML/Cloud in Last 5 Years ($USD)", min_value=0, max_value=50000, value=1000)

# Full country list based on training data
country = st.selectbox("Country You Reside In", [
    'France', 'Other', 'Australia', 'United States of America',
    'Italy', 'Brazil', 'Argentina', 'Japan', 'Canada', 'India',
    'Colombia', 'Pakistan', 'Mexico', 'Turkey', 'Nigeria', 'Tunisia',
    'Philippines', 'South Korea', 'Peru',
    'Iran, Islamic Republic of...', 'Russia', 'Bangladesh', 'Israel',
    'Kenya'])

job_title = st.selectbox("Current Role", [
    'Data Scientist', 'Software Engineer', 'Research Scientist', 'Developer Advocate',
    'Data Analyst (Business, Marketing, Financial, Quantitative, etc)', 'Data Engineer', 'Other',
    'Machine Learning/ MLops Engineer', 'Engineer (non-software)', 'Teacher / professor',
    'Manager (Program, Project, Operations, Executive-level, etc)', 'Statistician',
    'Data Administrator', 'Data Architect'
])
industry = st.selectbox("Industry of Current Employer", [
    'Online Service/Internet-based Services', 'Insurance/Risk Assessment', 'Government/Public Service',
    'Computers/Technology', 'Accounting/Finance', 'Academics/Education', 'Non-profit/Service',
    'Other', 'Medical/Pharmaceutical', 'Marketing/CRM', 'Manufacturing/Fabrication',
    'Energy/Mining', 'Broadcasting/Communications', 'Retail/Sales', 'Shipping/Transportation'
])
ml_incorporated = st.selectbox("Does Your Employer Use ML Methods?", [
    'We recently started using ML methods (i.e., models in production for less than 2 years)',
    'We have well established ML methods (i.e., models in production for more than 2 years)',
    'We are exploring ML methods (and may one day put a model into production)',
    'I do not know',
    'We use ML methods for generating insights (but do not put working models into production)',
    'No (we do not use ML methods)'
])

# Build input DataFrame
input_data = pd.DataFrame({
    "Q11 How many years have you been writing code and/or programming?": [years_coding],
    "Q16 How many years have you used machine learning methods?": [years_ml],
    "Q30 Approximately how much money have you spent on machine learning and/or cloud computing services at home or at work in the past 5 years ($USD)?": [money_spent],
    "Q4 Country you currently reside in?": [country],
    "Q23 Select title most similar to your current role": [job_title],
    "Q24 Industry of current employer/contract": [industry],
    "Q27 Does your current employer incorporate machine learning methods into their business?": [ml_incorporated]
})

# One-hot encode input to match model
input_data_encoded = pd.get_dummies(input_data)
model_columns = model.feature_names_in_

# Add missing columns
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns
input_data_encoded = input_data_encoded[model_columns]

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_data_encoded)[0]
    st.success(f"Estimated Yearly Compensation: ${prediction:,.2f}")
    
st.markdown("---")
st.markdown(
    "<small> Application Created By Philip Klim</small>",
    unsafe_allow_html=True)
