import streamlit as st

st.title("Simple Test")
st.write("Testing for black rectangle")

# Test navigation
page = st.selectbox("Page", ["Page 1", "Page 2"])

if page == "Page 1":
    st.write("This is page 1")
else:
    st.write("This is page 2")
