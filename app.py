#Imported all the necessary libraries first(Its my first time using this streamlit named library so i ad to look up some of the functions i have used)
import streamlit as st
import joblib 
import numpy as np 
import pandas as pd

#Configured the page
st.set_page_config(page_title="ChurnGuard AI")

#loaded the pickle files
scaler = joblib.load("scaler.pkl")
model = joblib.load("best_churn_model.pkl")

#header thingy and added a small description 
st.title("ChurnGuard AI")
st.markdown(" This is a high grade retention analytics for your customers...")
st.write("Enter all the customer stats below or you can also upload a batch file to predict the risk using our Support Vector Machine(SVM) model")
st.divider()

# Sidebad added too
with st.sidebar:

    st.header("About ChurnGuard")
    st.markdown("""
    **This was built during the horizon.**
    
    This application bridges the gap between raw data and very doable business intel. 
    
    **The Engine:**
    I powered this ai by a support vector classifier(SVC).
    """)
    
    st.divider()
    threshold = 0.75
#Built about 2 tabs one for the individual analysis and one for multiple or like batch system
tab1, tab2 = st.tabs(["Individual Analysis", "Batch Processing"]) 

#LEts work on the tab one first of all
with tab1:
    st.subheader("Single Customer Prediction(for trial)")
    st.info("Input a customer's current data/metrics  to get an instant real-time risk assessment of whether they will Churn or simply discontinue.")
    col_age, col_gender, col_tenure = st.columns(3)
    with col_age:
        age = st.number_input("Customer Age",min_value=18, max_value=100,  value=30)

    with col_gender:
        gender = st.selectbox("Customer Gender", ["Male","Female"])

    with col_tenure:
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=130, value=10)



    monthlycharge = st.number_input("Monthly Charge ($)", min_value=30, max_value=150, value=70)

#This logic is for this button to find the final risk it simplly like takes your input and feeds to the scaler pickle we made earlier 
    if st.button("Analyze Risk"):
        if gender == "Female":
            gender_selected = 1
        else:
            gender_selected = 0
        X = np.array([[age, gender_selected, tenure, monthlycharge]])
        X_scaled = scaler.transform(X)

        # to get the yes/no insted of 1 and 0
        probs = model.predict_proba(X_scaled)[0]
        churn_prob = probs[1]
       



        # then compare the threshhold and print those results for the user to see in the form of the markdowns
        if churn_prob >= threshold:
            st.error("Result: HIGH RISK")

            st.write(f"**Confidence Score:** {churn_prob}%")
            st.write("This customer shows strong attrition signals so immediate retention action is highly recommended.")
        else:
            st.success("Result: LOW RISK")


            retention_score = round((1 - churn_prob) * 100, 1)
            st.write("**Retention Probability:** ", str(retention_score), "%")
            st.write("This customer's profile does aligns with stable, long-term users so no need for specific action .")

#now fianlly wokring in the tab 2 for teh batch processing which is a bit complcated
with tab2:
    st.subheader("Bulk Customer Analysis")

    #THIS IS MOST IMPORTANT PART SINCE IT IS EXACTLY HOW ITS GOING TO BE INPUTED TO THE MODELLL alr
    st.write("Upload a CSV with these exact columns: `Age`, `Gender`, `Tenure`, `MonthlyCharges`.")
    #This new  streamlit makes it much more easier to uplod these files here
    uploaded_file = st.file_uploader("Drop your comma separted values here", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            processed_df = df.copy()
            processed_df['Gender'] = processed_df['Gender'].map({'Female': 1,'Male': 0})
           
            features = processed_df[['Age',  'Gender','Tenure', 'MonthlyCharges']]
            
            X_scaled = scaler.transform(features)
            batch_probs = model.predict_proba(X_scaled)[:,1] 
            
            # COnvert the probabilities to percentage so its human readable
            df['Churn_Risk_Score'] = (batch_probs * 100).round(1).astype(str) + '%'

        # Assign risk labels to them
            df['Prediction'] = 'Low Risk'
            df.loc[batch_probs >= threshold, 'Prediction'] = 'High Risk'    
           
            st.success(" Batch Analysis has been completed! BELOW are the results for you to see:")

           
            res_col1, res_col2 = st.columns(2)
            churn_count = int((batch_probs >= threshold).sum())
            churn_percentage = round((churn_count / len(df)) * 100, 1)
            
            res_col1.metric("Total Customers Analyzed", len(df))
            res_col2.metric("At-Risk Detected", churn_count, delta=str(churn_percentage) + "%", delta_color="inverse")
            st.dataframe(df)

            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Detailed Report in csv from the button below",data=csv_output,file_name="horizon_batch_predictions.csv",
                mime="text/csv")
            
#Ofc we need to have an except when we have the try statement              
        except:
            st.error("Error processing file. Please make sure the CSV has the exact columns: Age, Gender, Tenure, MonthlyCharges NOTE:THIS IS A MUST SO THAT THIS WORKS. FOR TESTING YOU CAN USE THE sample_dataset i have given in github repo")

#PLEASE keep in mind THAT THE model AND THE scaler pickel files ARE NOT INCLUDED SO YOU NEED TO HAVE THEM IN THE SAME DIRECTORY AS THIS APP.PY FILE FOR IT TO WORK PROPERLY then alsoo note that I 