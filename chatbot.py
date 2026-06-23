import json
import pickle
import random
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load model
model = pickle.load(
    open("models/chatbot_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("models/tfidf_vectorizer.pkl", "rb")
)

with open("dataset/intents.json", "r") as file:
    intents = json.load(file)

# Preprocessing
def preprocess(text):

    tokens = nltk.word_tokenize(text.lower())

    cleaned = []

    for word in tokens:
        if word.isalpha() and word not in stop_words:
            cleaned.append(
                lemmatizer.lemmatize(word)
            )

    return " ".join(cleaned)

# Chat function
def get_response(user_input):

    # Rule-based greetings
    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if user_input.lower() in greetings:
        return "Hello! How can I assist you today?"

    processed = preprocess(user_input)

    vector = vectorizer.transform([processed])

    probabilities = model.predict_proba(vector)[0]

    predicted_index = probabilities.argmax()

    prediction = model.classes_[predicted_index]

    confidence = probabilities[predicted_index]

    print("Intent:", prediction)
    print("Confidence:", confidence)

    if confidence < 0.30:
        return "Sorry, I do not understand your question."

    for intent in intents["intents"]:

        if intent["tag"] == prediction:

            return random.choice(
                intent["responses"]
            )

    return "Sorry, I do not understand your question."