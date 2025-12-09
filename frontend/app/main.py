import json

import streamlit as st
import requests
import style as style

# Constants
BACKEND_URL = "http://backend:4242"
NUMBER_OF_MESSAGES_TO_DISPLAY = 20

def create_request(ask_input):
    try:
        body = {
            "data": ask_input
        }
        response = requests.post(BACKEND_URL + "/create", json=body).json()
        st.write("Answer :")
        st.json(response)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the Streamlit API documentation: {str(e)}")
    return None

def main():
    st.title("Software Crew IA")

    # Insert custom CSS for glowing effect
    st.markdown(style.streamlit_style, unsafe_allow_html=True)

    ask_input = st.text_input(label="Ask me about use case creation")
    if ask_input:
        st.write("You entered: ", ask_input)
        create_request(ask_input)

if __name__ == "__main__":
    main()