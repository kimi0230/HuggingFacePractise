import streamlit as st
from transformers import pipeline

st.title("GPT-2 Text Generation")

pipe = pipeline('text-generation', model='gpt2')
text = st.text_area('Enter some text')

if text:
    out = pipe(text, max_length=50, do_sample=True)
    generated_text = out[0]['generated_text'].strip()
    st.write("Generated Text:")
    st.write(generated_text)
