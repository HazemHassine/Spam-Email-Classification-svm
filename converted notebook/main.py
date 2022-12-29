from logging import PlaceHolder
import streamlit as st
import pickle as pkl
from Classifying_spam_emails_SVM import encode_text, prediction
import pandas as pd

encode_text = pkl.load(open("encode.pkl","rb"))
prediction = pkl.load(open("pred.pkl","rb"))
model = pkl.load(open("model.pkl", "rb"))

st.write("### Check if this email is spam or ham")
input = st.text_area(label="copy and paste the email here or write it")
btn = st.button("Check spam or ham")
if btn:
    result = prediction(model, input, encode=True)
    result = "Spam" if result[0] == 1 else "Ham"
    st.header(result)
    if result == "Ham":
        st.write("""
            <style>
            .appview-container {
                background-color: #62D932
                }
            </style>
        """, 
        unsafe_allow_html=True)
    else:
        st.write("""
            <style>
            .appview-container {
                background-color: #db1b0d
                }
            </style>
        """, 
        unsafe_allow_html=True)