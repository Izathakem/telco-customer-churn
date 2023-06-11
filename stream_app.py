import pickle
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


def main():
    add_selectbox = st.sidebar.selectbox(
        "Would you like to predict or see the data visualization",
        ("Prediction", "VizData"))
    st.sidebar.info('Click here to see more detail about customer churn')
    st.title("Predicting Customer Churn")
    
    if add_selectbox == 'Prediction':
        gender = st.selectbox('Gender:', ['male', 'female'])
        senior_citizen = st.selectbox('Customer is a senior citizen:', [0, 1])
        partner = st.selectbox('Customer has a partner:', ['yes', 'no'])
        dependents = st.selectbox('Customer has dependents:', ['yes', 'no'])
        phone_service = st.selectbox('Customer has phone service:', ['yes', 'no'])
        multiple_lines = st.selectbox('Customer has multiple lines:', ['yes', 'no', 'no_phone_service'])
        internet_service = st.selectbox('Customer has internet service:', ['dsl', 'no', 'fiber_optic'])
        online_security = st.selectbox('Customer has online security:', ['yes', 'no', 'no_internet_service'])
        online_backup = st.selectbox('Customer has online backup:', ['yes', 'no', 'no_internet_service'])
        device_protection = st.selectbox('Customer has device protection:', ['yes', 'no', 'no_internet_service'])
        tech_support = st.selectbox('Customer has tech support:', ['yes', 'no', 'no_internet_service'])
        streaming_tv = st.selectbox('Customer has streaming TV:', ['yes', 'no', 'no_internet_service'])
        streaming_movies = st.selectbox('Customer has streaming movies:', ['yes', 'no', 'no_internet_service'])
        contract = st.selectbox('Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
        paperless_billing = st.selectbox('Customer has paperless billing:', ['yes', 'no'])
        payment_method = st.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check', 'mailed_check'])
        tenure = st.number_input('Number of months the customer has been with the current telco provider:', min_value=0, max_value=240, value=0)
        monthly_charges = st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
        total_charges = tenure * monthly_charges
        output = ""
        output_prob = ""
        
        input_dict = {
            "gender": gender,
            "seniorcitizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "phoneservice": phone_service,
            "multiplelines": multiple_lines,
            "internetservice": internet_service,
            "onlinesecurity": online_security,
            "onlinebackup": online_backup,
            "deviceprotection": device_protection,
            "techsupport": tech_support,
            "streamingtv": streaming_tv,
            "streamingmovies": streaming_movies,
            "contract": contract,
            "paperlessbilling": paperless_billing,
            "paymentmethod": payment_method,
            "tenure": tenure,
            "monthlycharges": monthly_charges,
            "totalcharges": total_charges
        }

        if st.button("Prediction"):
            X = dv.transform([input_dict])
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
        
        st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))

    if add_selectbox == 'VizData':
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        services = ['PhoneService', 'InternetService', 'TechSupport', 'StreamingTV']
        selected_service = st.selectbox("Churn reason:", services)

        def churn_rate(service):
            fig = plt.figure(figsize=(10, 6))
            svc_types = df.groupby(service)['Churn'].value_counts(normalize=True).unstack()
            svc_types.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title(service)
            plt.tight_layout()
            st.pyplot(fig)

        churn_rate(selected_service)


if __name__ == '__main__':
    main()
