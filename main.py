import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import streamlit as st

# Load intents JSON
with open("intents.json", "r") as f:
    data = json.load(f)

patterns = []
tags = []
responses = {}

# Extract patterns, tags, responses
for intent in data["intents"]:
    for p in intent["patterns"]:
        patterns.append(p)
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Convert text to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train classifier
clf = LinearSVC()
clf.fit(X, tags)

# Predict response
def get_response(msg):
    text_vect = vectorizer.transform([msg])
    tag = clf.predict(text_vect)[0]
    return random.choice(responses[tag]), tag

# Streamlit UI
st.title("ML Chatbot – No TensorFlow Needed 🎉")

user_input = st.text_input("Say something:")

if user_input:
    reply, tag = get_response(user_input)
    st.write(f"**Bot ({tag}):** {reply}")
