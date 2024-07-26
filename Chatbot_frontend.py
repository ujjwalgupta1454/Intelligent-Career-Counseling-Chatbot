import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import nltk
nltk.download('stopwords')
nltk.download("punkt")
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import load_model

from PIL import Image

# <------------------------------------------------------------- Functions ----------------------------------------------------------------------------------->
def clean_up_sentence(sentence):
    # Process and normalize input sentence
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return ' '.join(sentence_words)

def bow(sentence, words, show_details=True):
    # Convert input sentence to bag-of-words representation
    sentence_words = clean_up_sentence(sentence).split()
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(bow_input, model):
    # Predict class label from the bag-of-words input using the model
    prediction = model.predict(np.array([bow_input]))[0]
    predicted_class = np.argmax(prediction)
    return predicted_class

def chatbot_response(predicted_class):
    # Generate response based on predicted class label
    for intent in intents['intents']:
        if intent['tag'] == classes[predicted_class]:
            response = np.random.choice(intent['responses'])
            return response
    return "Sorry, I didn't understand that."

# <---------------------------------------------------------- Page Configuration ----------------------------------------------------------------------------->
im = Image.open('bot.jpg')
st.set_page_config(layout="wide", page_title="Intelligent Career Counseling Chatbot", page_icon=im)

# <---------------------------------------------------------- Main Header ------------------------------------------------------------------------------------->
st.markdown(
    """
    <div style="background-color: #FF8C00 ; padding: 10px">
        <h1 style="color: brown; font-size: 48px; font-weight: bold">
           <center> <span style="color: black; font-size: 64px">I</span>ntelligent <span style="color: black; font-size: 64px">C</span>areer <span style="color: black; font-size: 64px">C</span>ounseling <span style="color: black; font-size: 64px">C</span>hatbot </center>
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# <========================================================= Importing Data Files  ====================================================================>
with open('intents3.json', 'r') as file:
    intents = json.load(file)
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

# <--------------------------hide the right side streamlit menu button --------------------------------->
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# <============================================================= Sidebar ================================================================================> 
with st.sidebar:
    st.title('🤗💬 Intelligent Career Counseling Chatbot')
    st.markdown('''
    ## About
    Welcome to Intelligent Career Counseling Chatbot, your personalized assistant designed to provide career guidance and recommendations based on your interests and goals. Leverage our advanced NLP capabilities to effectively navigate your career path.

    ## Contributing 🤝
    We welcome contributions to enhance the project and make it even more effective. If you have any suggestions or bug fixes, or if you would like to propose new features, please contact me at  
    Email: ujjwalgupta1454@gmail.com  
    We appreciate your contributions!
    ''')

# <============================================================= Initializing Session State ==========================================================>
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm an AI Career Counselor, How may I help you?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# <================================================== Function for taking user provided prompt as input ================================================>
def get_text():
    # Retrieve user input from text box
    input_text = st.text_input("You: ", key="input")
    return input_text

styl = f"""
<style>
    .stTextInput {{
    position: fixed;
    bottom: 20px;
    z-index: 20;
    font-size: 20px; /* Increase font size */
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

# <================================================ Loading The Model ===============================================================>
model = load_model('chatbot_model.h5')

# <============================== Function for taking user prompt as input followed by producing AI generated responses ============> 
def generate_response(prompt):
    # Generate chatbot response for given input
    bow_input = bow(prompt, words, show_details=False)
    predicted_class = predict_class(bow_input, model)
    response = chatbot_response(predicted_class)
    return response

# <====================== Conditional display of AI generated responses as a function of user provided prompts =====================================>
with response_container:
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i))

with input_container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("You: ", key="input")
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
