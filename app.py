# importing necessary libraries
import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import plotly.graph_objects as go
import time
import os

#page set up
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    # Define the path where NLTK data should be stored
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    
    # Check if NLTK data already exists
    if not os.path.exists(nltk_data_path):
        # If it doesn't exist, download the data
        for resource in ['punkt_tab', 'stopwords', 'wordnet']:  
            nltk.download(resource, quiet=True)
    
    # Set the NLTK data path
    nltk.data.path.append(nltk_data_path)

# Call the function at the start of your script
download_nltk_data()

# Load the model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = load('best_model_RandomForest-200.pkl')
        vectorizer = load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, vectorizer = load_model_and_vectorizer()

# Text cleaning function
@st.cache_data
def clean_text(text):
    try:
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        cleaned_text = ' '.join(tokens)
        return cleaned_text
    except Exception as e:
        st.error(f"Error in clean_text: {str(e)}")
        return text # Return original text if cleaning fails

# Function to classify a single email
def classify_email(email):
    cleaned_email = clean_text(email)
    vectorized_email = vectorizer.transform([cleaned_email])
    prediction = model.predict(vectorized_email)[0]
    probability = model.predict_proba(vectorized_email)[0]
    return "Spam" if prediction == 1 else "Non-spam", probability

# Streamlit app

st.title("📧 Email Spam Classifier")

st.markdown("""
This app uses a machine learning model to classify emails as spam or non-spam (ham).
You can either enter an email manually or upload a CSV file containing multiple emails.
""")

# Sidebar
st.sidebar.header("Options")
input_method = st.sidebar.radio("Choose input method:", ("Manual Input", "CSV Upload"))

if input_method == "Manual Input":
    st.header("Manual Email Input")
    email_input = st.text_area("Enter the email content here:", height=200)
    
    if st.button("Classify Email"):
        if email_input:
            with st.spinner("Classifying..."):
                result, probability = classify_email(email_input)
                time.sleep(1)  # Simulate processing time
            
            st.subheader("Classification Result:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", result)
            with col2:
                confidence = probability[1] if result == "Spam" else probability[0]
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Confidence in {result} Classification"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig)
        else:
            st.warning("Please enter an email to classify.")

else:
    st.header("CSV File Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'Email' not in df.columns:
            st.error("The CSV file must contain an 'Email' column.")
        else:
            st.success("File successfully uploaded!")
            
            if st.button("Classify Emails"):
                with st.spinner("Classifying emails..."):
                    results = []
                    for email in df['Email']:
                        result, probability = classify_email(email)
                        confidence = probability[1] if result == "Spam" else probability[0]
                        results.append({'Email': email + '...', 'Classification': result, 'Confidence': f"{confidence:.2%}"})
                    
                    results_df = pd.DataFrame(results)
                    st.subheader("Classification Results:")
                    st.dataframe(results_df.head(10))
                    
                    # total mail analysed
                    st.write(f'Total email classified: {len(results_df)}')

                    # count of each unique value in the 'Classification' column
                    classification_counts = results_df['Classification'].value_counts()

                    # Assigning the counts to variables based on the label
                    Non_spam_count = classification_counts.get('Non-spam', 0)  # Default to 0 if 'Non-spam' not found
                    spam_count = classification_counts.get('Spam', 0)  # Default to 0 if 'Spam' not found

                    # Display the counts
                    st.write(f"Non-spam count: {Non_spam_count}")
                    st.write(f"Spam count: {spam_count}")

                    # Download link for results
                    col1, col2, col3 = st.columns([1.5, 1.5, 3])
                    csv = results_df.to_csv(index=False)
                    html = results_df.to_html()

                    with col1:
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="email_classification_results.csv",
                            mime="text/csv",
                        )

                    with col2:
                        st.download_button(
                            label="Download Results as html",
                            data=html,
                            file_name="email_classification_results.html",
                            mime="text/html",
                        )
                    # with col3:
                    #     write('')

                    # Visualization
                    import plotly.express as px
                    fig = px.pie(results_df, names='Classification', title='Email Classification Distribution')
                    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Created by Arthur_Techy")