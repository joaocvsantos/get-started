import streamlit as st

st.title("🎈 My new app, version 2")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

value = st.slider('Select a value', min_value=0, max_value=100, value=50)
