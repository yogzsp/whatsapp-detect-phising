import argparse
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import base64
import requests
import json

def sendWhatsapp(message):
    # URL endpoint
    url = "https://mada.yogzsp.site/api/OTA4MTQxMDU5MzE3OTI2NDkwa3l3MDgwMjg2ODRYOTY4ODA1Mzg4RA/chat"

    # Body message to send in the POST request
    msg = {
        "msg": message,
        # Add other key-value pairs as needed
    }

    # Convert the body message to JSON format
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(msg))

    # Check if the request was successful
    if response.status_code == 200:
        print("Request was successful.")
        print("Response:", response.json())
    else:
        print(f"Request failed with status code {response.status_code}.")
        print("Response:", response.text)


# Load the dataset
data = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')

# Verify URL column
print(data['URL'].head())
print(data['URL'].isnull().sum())

# Cleaning URLs
def clean_url(url):
    url = re.sub(r'http://|https://|www\.', '', url)  # Simplified URL cleaning
    url = re.sub(r'\W', ' ', url)
    url = re.sub(r'\s+', ' ', url).strip()
    return url

# Apply cleaning function
data['cleaned_url'] = data['URL'].apply(clean_url)

# Check cleaned URLs
print(data['cleaned_url'].head())
print(data['cleaned_url'].isnull().sum())

# Remove rows with empty 'cleaned_url'
data = data[data['cleaned_url'] != '']

# Ensure cleaned URL column has data
if data['cleaned_url'].empty:
    raise ValueError("The 'cleaned_url' column is empty.")

# Vectorize URLs
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['cleaned_url'])
y = data['label']  # Make sure this matches your column name

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean()}")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Phishing', 'Phishing'], yticklabels=['Not Phishing', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC and AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Save the model
joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Function to classify new URLs
def classify_url(url):
    model = joblib.load('naive_bayes_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    cleaned_url = clean_url(url)
    features = vectorizer.transform([cleaned_url])
    return model.predict(features)

# Command-line argument parsing
def main():
    parser = argparse.ArgumentParser(description='Phishing URL Classifier')
    parser.add_argument('url', type=str, help='URL to classify')
    args = parser.parse_args()
    
    url_baru = args.url
    classification = classify_url(url_baru)[0]
    print(f"URL {url_baru} diklasifikasikan sebagai: {classification}")
    if classification:
        sendWhatsapp(f"URL {url_baru} diklasifikasikan sebagai link phising")
# Run the main function if this script is executed
if __name__ == "__main__":
    main()
