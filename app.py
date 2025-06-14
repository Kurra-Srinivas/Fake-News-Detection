import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert")
    tokenizer = BertTokenizer.from_pretrained("./fine_tuned_bert")
    return model, tokenizer

model, tokenizer = load_model()

st.title("Fake News Detector")
st.write("Enter a news article to classify it as **Fake** or **Real**.")
user_input = st.text_area("News Article", height=200)

if st.button("Classify"):
    if user_input.strip():
        inputs = tokenizer(user_input, padding=True, truncation=True, max_length=512, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
        label = "Real" if prediction == 1 else "Fake"
        st.success(f"Prediction: **{label}**")
    else:
        st.error("Please enter a news article.")

st.write("Built with BERT and Streamlit.")