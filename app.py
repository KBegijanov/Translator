import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-en-ru"
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

st.title("AI-переводчик с использованием Hugging Face и Streamlit")

text_input = st.text_area("Введите текст для перевода:", value="", height=200)

if st.button("Перевести"):
    tokenized_text = tokenizer(text_input, return_tensors="pt")
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)
    st.write(translated_text[0])
