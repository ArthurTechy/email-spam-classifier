# Email Spam Classifier

## Overview
This Streamlit application implements a machine learning model to classify emails as spam or non-spam (ham). Users can analyze the sentiment of emails through manual input or by uploading a CSV file containing multiple emails.

## Features
- Spam prediction for individual emails
- Batch processing of emails via CSV upload
- Interactive web interface using Streamlit
- Visualization of classification results using Plotly
- Downloadable CSV of classification results for batch processing

## Dataset
- **Source**: Kaggle
- **File**: emails.csv 
- **Link**: [Spam email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset/data)

## Model
- **Algorithm**: Random Forest Classifier
- **Parameters**: n_estimators = 200, random_state = 42

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier

# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

To run the Streamlit app:

```bash
streamlit run app.py
```

## File Structure
- `app.py`: Main Streamlit application
- `best_model_RandomForest-200.pkl`: Trained Random Forest model
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer
- `requirements.txt`: List of required Python packages

## Dependencies
- streamlit
- pandas
- nltk
- scikit-learn
- plotly
- joblib

## Features in Detail

1. **Manual Input**: Users can enter a single email for spam classification.
2. **CSV Upload**: Batch processing of emails from a CSV file.
3. **Text Preprocessing**: Includes lowercasing, HTML tag removal, punctuation removal, tokenization, stopword removal, and lemmatization.
4. **Visualization**: 
   - Gauge chart for confidence in classification for single emails
   - Pie chart for distribution of spam vs. non-spam in batch processing
5. **Results Download**: Option to download batch processing results as a CSV or html file.

## How It Works

1. The app loads a pre-trained Random Forest model and TF-IDF vectorizer.
2. User input (manual or CSV) is cleaned and preprocessed.
3. The model predicts whether each email is spam or non-spam.
4. Results are displayed with classification and confidence level.
5. Visualizations provide additional insights into the classifications.

## Future Improvements
- Implement more advanced NLP techniques like BERT or transformers for potentially better accuracy
- Add support for multi-class spam classification
- Integrate email header analysis for more comprehensive spam detection

## Contributing
Contributions to improve the Email Spam Classifier are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

Please ensure your code adheres to our coding standards and include tests for new features.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or feedback about this project, please reach out to:

- Name: Chiezie Arthur Ezenwaegbu
- GitHub: @Arthur_Techy

You can also open an issue in the GitHub repository if you encounter any problems or have feature requests.