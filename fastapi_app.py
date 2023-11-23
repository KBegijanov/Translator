from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Helsinki-NLP/opus-mt-en-ru"
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/translator/")
def translator():
    tokenized_text = tokenizer("Hi! How are you?", return_tensors="pt")
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)
    return translated_text