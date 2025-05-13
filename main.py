# Step 1: Import Libraries and Load the Model
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

MAX_FEATURES = 10_000
MAX_LEN      = 500
INDEX_FROM   = 3        # imdb dataset offset for special tokens

@st.cache_resource(show_spinner=False)
def load_artifacts():
    word_index = imdb.get_word_index()
    model = tf.keras.models.load_model("simple_rnn_imdb.h5", compile=False)
    return word_index, model

word_index, model = load_artifacts()

def preprocess(text: str) -> np.ndarray:
    tokens  = text.lower().split()
    indices = [min(word_index.get(t, 2) + INDEX_FROM, MAX_FEATURES - 1)
               for t in tokens]
    return sequence.pad_sequences([indices], maxlen=MAX_LEN)

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
review = st.text_area('Movie Review')

if st.button("Classify"):
    if not review.strip():
        st.warning("Please type a review first.")
        st.stop()

    x      = preprocess(review)
    prob   = float(model.predict(x, verbose=0)[0][0])
    senti  = "Positive" if prob > 0.5 else "Negative"

    st.success(f"**Sentiment:** {senti}")
    st.info   (f"**Confidence:** {prob:.4f}")

