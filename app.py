import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("balanced_cyberbullying.csv")

# Lowercase text
df['Text'] = df['Text'].str.lower()

# ----------------------------
# TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['Text'])
y = df['oh_label']

# ----------------------------
# Train Model
# ----------------------------
model = LogisticRegression()
model.fit(X, y)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Cyberbullying Detection App")

st.write("Enter a message to check whether it is Cyberbullying or Not.")

user_input = st.text_area("Enter Text Here")

if st.button("Predict"):
    input_lower = user_input.lower()
    input_vector = vectorizer.transform([input_lower])
    prediction = model.predict(input_vector)

    if prediction[0] == 1:
        st.error("⚠️ Cyberbullying Detected")
    else:
        st.success("✅ Not Cyberbullying")