import json
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())

    cleaned = []

    for word in tokens:
        if word.isalpha() and word not in stop_words:
            cleaned.append(lemmatizer.lemmatize(word))

    return " ".join(cleaned)

# Load dataset
with open("dataset/intents.json", "r") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(preprocess(pattern))
        tags.append(intent["tag"])

# TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(patterns)
y = tags

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("=================")

print("Accuracy:", accuracy)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(cm)

disp.plot(xticks_rotation=90)

plt.title("Confusion Matrix")

plt.show()

# Save model
pickle.dump(
    model,
    open("models/chatbot_model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("models/tfidf_vectorizer.pkl", "wb")
)

print("\nModel saved successfully!")