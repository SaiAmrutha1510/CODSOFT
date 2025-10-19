# smart_chatbot_dt.py
import streamlit as st
import re
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

# --- RULE-BASED HANDLER ---
def rule_based_reply(user_input):
    user_input = user_input.lower().strip()

    if re.search(r"\bhi\b|\bhello\b|\bhey\b", user_input):
        return "Hello! How can I assist you today? 😊"
    elif "how are you" in user_input:
        return "I'm doing well, thanks for asking! How about you?"
    elif "your name" in user_input:
        return "I'm Chatnandu, your friendly assistant 🤖"
    elif "bye" in user_input:
        return "Goodbye! Have a great day! 👋"
    elif "thank" in user_input:
        return "You're welcome! 😊"
    return None  # fallback to ML if not matched

# --- TRAIN DECISION TREE ON BASIC INTENTS ---
training_texts = [
    "what is your purpose",
    "tell me a joke",
    "what can you do",
    "what's the time",
    "what's the date today",
    "can you help me",
    "who created you",
    "tell me something",
    "how do you work",
    "are you intelligent"
]
intent_labels = [
    "purpose", "joke", "ability", "time", "date",
    "help", "creator", "fact", "working", "intelligence"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_texts)
clf = DecisionTreeClassifier()
clf.fit(X, intent_labels)

# --- ML FALLBACK HANDLER ---
def ml_based_reply(user_input):
    x_test = vectorizer.transform([user_input])
    intent = clf.predict(x_test)[0]

    replies = {
        "purpose": "I'm here to assist and chat with you anytime!",
        "joke": "Why did the computer go to the doctor? Because it had a virus! 😂",
        "ability": "I can chat, tell jokes, share info, and more!",
        "time": "I can't see clocks, but you can check the bottom right corner ⏰",
        "date": "You can check your calendar — I don’t have real-time access yet!",
        "help": "Of course! Ask me anything you like.",
        "creator": "I was created by a cool developer like you!",
        "fact": "Did you know honey never spoils? 🍯",
        "working": "I use simple rules and a decision tree to answer you.",
        "intelligence": "Not yet super smart, but always improving!"
    }
    return replies.get(intent, "I'm not sure how to respond to that yet.")

# --- STREAMLIT UI ---
st.set_page_config(page_title="🌳 Smart Chatbot (Rule + Decision Tree)", layout="centered")
st.title("🤖 Smart Chatbot with Rule-Based Responses")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("You:")

if user_input:
    st.session_state.chat_history.append(("You", user_input))
    time.sleep(0.5)  # Add slight delay before bot replies

    rule_response = rule_based_reply(user_input)
    if rule_response:
        response = rule_response
    else:
        response = ml_based_reply(user_input)

    st.session_state.chat_history.append(("Chatnandu", response))

# Display full chat history with styled bubbles
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(
            f"""
            <div style='
                background-color: #dcf8c6;
                color: black;
                text-align: right;
                padding: 10px 15px;
                border-radius: 18px;
                margin: 10px 0 10px auto;
                max-width: 80%;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            '>
                <b>{sender}</b>: {message}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='
                background-color: #f1f0f0;
                color: black;
                text-align: left;
                padding: 10px 15px;
                border-radius: 18px;
                margin: 10px auto 10px 0;
                max-width: 80%;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            '>
                <b>{sender}</b>: {message}
            </div>
            """,
            unsafe_allow_html=True,
        )
