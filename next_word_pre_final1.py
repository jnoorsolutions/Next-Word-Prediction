# 📦 Libraries
import streamlit as st
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🎨 Streamlit UI Config
st.set_page_config(page_title="📖 Next Word Predictor | Hamlet LSTM", layout="centered")

# 🧠 Title
st.markdown("<h4 class='section-title'>🧠 Next Word Prediction using LSTM (Hamlet)</h4>", unsafe_allow_html=True)

# ✍️ Input Section
st.markdown("<div class='section-title'>✍️ Paste a Sentence Below for Prediction:</div>", unsafe_allow_html=True)
input_text = st.text_input("📌 Enter your sentence:")

# 🌈 Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        padding: 1rem;
    }
    .hamlet-box {
        font-family: 'Courier New', monospace;
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 10px;
        height: 300px;
        overflow-y: scroll;
        border: 1px solid #444;
        white-space: pre-wrap;
    }
    .highlight {
        background-color: yellow;
        color: black;
        font-weight: bold;
        padding: 0 2px;
        border-radius: 3px;
    }
    .section-title {
        color: teal;
        font-weight: 600;
        font-size: 20px;
        margin-top: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# 🔄 Load model and tokenizer
@st.cache_resource
def load_artifacts():
    model = load_model('next_word_lstm.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_artifacts()
max_sequence_len = model.input_shape[1] + 1  # input_length + 1

# 🧠 Prediction Function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if not token_list:
        return "⚠️ Word not in vocabulary", None

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)

    predicted_word_index = np.argmax(predicted, axis=1)[0]
    predicted_prob = round(float(predicted[0][predicted_word_index]) * 100, 2)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word, predicted_prob

    return None, None

# 📄 Load Hamlet Text from File
@st.cache_data
def load_hamlet_text():
    with open("shakespeare_files.txt", "r", encoding="utf-8") as f:
        return f.read()

hamlet_text_raw = load_hamlet_text()

# 🔦 Highlight predicted word in Hamlet text
#def highlight_word_in_text(text, word_to_highlight):
#    if not word_to_highlight or word_to_highlight.strip() == "":
#        return text  # nothing to highlight
#    pattern = re.compile(rf'\b({re.escape(word_to_highlight)})\b', re.IGNORECASE)
#    highlighted_text = pattern.sub(r"<span class='highlight'>\1</span>", text, count=1)
#    return highlighted_text

def highlight_word_in_text(text, sentence_fragment, predicted_word):
    if not sentence_fragment or not predicted_word:
        return text

    # Escape special characters for regex safety
    escaped_sentence = re.escape(sentence_fragment.strip())
    escaped_predicted = re.escape(predicted_word.strip())

    # Look for the sentence in the Hamlet text
    match = re.search(escaped_sentence + r'(.*?)\b(' + escaped_predicted + r')\b', text, re.IGNORECASE | re.DOTALL)
    
    if match:
        # Only replace the predicted word that appears right after the sentence
        start, end = match.span(2)
        highlighted = (
            text[:start] +
            f"<span class='highlight'>{text[start:end]}</span>" +
            text[end:]
        )
        return highlighted

    # Fallback: highlight the first occurrence (legacy)
    pattern = re.compile(rf'\b({escaped_predicted})\b', re.IGNORECASE)
    return pattern.sub(r"<span class='highlight'>\1</span>", text, count=1)


predicted_word = ""
confidence = None

if input_text:
    # OOV words check
    input_words = input_text.lower().split()
    #oov_words = [word for word in input_words if word not in tokenizer.word_index]
    #if oov_words:
    #    st.warning(f"⚠️ Words not in vocabulary: {', '.join(oov_words)}")

    with st.spinner("🔍 Predicting next word..."):
        predicted_word, confidence = predict_next_word(model, tokenizer, input_text, max_sequence_len)

    if predicted_word:
        st.success(f"✅ Predicted Next Word: **{predicted_word}**")
        if confidence:
            st.caption(f"🔎 Confidence: {confidence}%")
    else:
        st.error("❌ Could not predict. Please try a different phrase.")
else:
    st.info("📌 Paste or type a sentence above from the Hamlet text.")

# 🖍 Highlighted Hamlet Text
highlighted_hamlet = highlight_word_in_text(hamlet_text_raw,input_text.strip(), predicted_word)

st.markdown("<div class='section-title'>📘 Hamlet Text with Highlight:</div>", unsafe_allow_html=True)
st.markdown(f"<div class='hamlet-box'>{highlighted_hamlet}</div>", unsafe_allow_html=True)

# 📚 Vocabulary Source Reference
st.markdown("""
<hr>
<h5 style='color: grey;'>📚 Vocabulary Source:</h5>
<p style='font-size: 15px;'>
This model is trained on <b>Shakespeare's Hamlet</b>, primarily on passages like:
<blockquote>
"To be, or not to be: that is the question:<br>
Whether 'tis nobler in the mind to suffer<br>
The slings and arrows of outrageous fortune..."
</blockquote>
Please write in a similar literary style for best results.
</p>
<hr>
""", unsafe_allow_html=True)
# 📖 Show Sample Vocabulary
#st.markdown("<div class='section-title'>📖 Sample Vocabulary</div>", unsafe_allow_html=True)
#with st.expander("🔍 Show top 50 words used for training"):
#    top_words = list(tokenizer.word_index.keys())[:50]
#    st.code(", ".join(top_words))
