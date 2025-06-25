# 🧠 Next Word Prediction using LSTM (Shakespeare's Hamlet)

This project is a **Streamlit-based web application** that predicts the **next word** in a given sentence using an LSTM model trained on Shakespeare's *Hamlet*. It allows users to interactively test the model and view predicted words highlighted within the original Hamlet text.

---

## 🚀 Features

- 📜 **Hamlet Text Integration**: Scrollable, read-only view of Hamlet for context.
- 🔤 **Next Word Prediction**: Predicts the next likely word using an LSTM model.
- ✍️ **User Input**: Enter or paste sentences from Hamlet into the input box.
- 🎯 **Vocabulary Detection**: Warns about out-of-vocabulary (OOV) words.
- 🌟 **Highlighting**: Highlights the predicted word within Hamlet's text (if found).

---

## 📦 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: [TensorFlow](https://www.tensorflow.org/) LSTM
- **Tokenizer**: Keras `Tokenizer`
- **Data**: Shakespeare's *Hamlet* (Public Domain)

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hamlet-lstm-predictor.git
cd hamlet-lstm-predictor
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run next_word_pre_final1.py
```

Then open the local server URL in your browser (usually [http://localhost:8501](http://localhost:8501)).

---

## 📁 Project Structure

```
📦 hamlet-lstm-predictor
├── next_word_pre_final1.py # Main Streamlit application
├── next_word_lstm.h5       # Trained LSTM model
├── tokenizer.pickle        # Tokenizer object
├── shakespeare_files.txt   # Raw Hamlet text used for training/reference
├── requirements.txt        # Required Python libraries
└── README.md               # Project documentation
```

---

## 🧪 Example Usage

1. Scroll through Hamlet text for reference.
2. Copy any phrase or write your own in the input field and click Text Editor.
3. View the **predicted next word** and **highlighted result** in the Hamlet text.

---

## 📊 Model Details

- Type: LSTM (Long Short-Term Memory Neural Network)
- Framework: TensorFlow / Keras
- Trained Text: Shakespeare's *Hamlet*
- Input Format: Padded sequences of tokenized text

---

## 📌 License

This project is licensed under the MIT License. You are free to use, modify, and distribute it with proper attribution.

---

## 🙌 Acknowledgements

- William Shakespeare - *Hamlet* (Public Domain)
- Streamlit for rapid UI development
- TensorFlow/Keras for model training and deployment

---

## 💡 Future Enhancements

-

---

Made with ❤️ by [Your Name](https://github.com/your-username)

