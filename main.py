# import random
# import json
# import pickle
# import numpy as np
# import nltk
# import speech_recognition as sr
#
# nltk.download('punkt')
#
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.models import load_model
#
# lemmatizer = WordNetLemmatizer()
# # Initialize Speech Recognition
# recognizer = sr.Recognizer()
#
# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('chatbotmodel.h5')
#
# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
#     return sentence_words
#
# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)
#
# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
#     return return_list
#
# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "Sorry, I couldn't understand that. Could you please rephrase your question or just let me know what kind of clinical help we can provide?"
#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if i['tag'] == tag:
#             result = random.choice(i['responses'])
#             return result
#
# def listen_for_audio():
#     with sr.Microphone() as source:
#         print("| Listening... (speak your query)")
#         audio = recognizer.listen(source)
#         try:
#             print("| Recognizing...")
#             query = recognizer.recognize_google(audio)
#             print(f"| You (spoken): {query}")
#             return query
#         except sr.UnknownValueError:
#             print("| Sorry, I couldn't understand that.")
#             return ""
#         except sr.RequestError:
#             print("| Sorry, there was an issue with the speech recognition service.")
#             return ""
#
# # Main chatbot loop
# # Main chatbot loop
# print("|=================== Welcome to Clinic Chatbot! ======================|")
# print("|============================== Feel Free ============================|")
# print("|================================== To ===============================|")
# print("|=============== Ask your any query about our Clinic ================|")
#
# # Ask the user to choose an input method only once
# while True:
#     print("| Choose an input method: (1) Text, (2) Voice")
#     choice = input("| Your Choice: ")
#     if choice in ["1", "2"]:
#         break
#     print("| Invalid choice. Please select 1 or 2.")
#
# while True:
#     if choice == "1":
#         message = input("| You (text): ")
#     elif choice == "2":
#         message = listen_for_audio()
#
#     if not message:
#         continue
#
#     if message.lower() == "bye" or message.lower() == "goodbye":
#         print("|===================== The Program End here! =====================|")
#         break
#
#     # Get the intents for the message
#     ints = predict_class(message)
#
#     # Get the appropriate response based on the predicted intent
#     res = get_response(ints, intents)
#     print("| Your Assistant:", res)
#

import random
import json
import pickle
import numpy as np
import nltk
import speech_recognition as sr
import pyttsx3

# Download required nltk data
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
recognizer = sr.Recognizer()

# Load data and models
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    """Tokenizes and lemmatizes the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Creates a bag of words from the input sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """Predicts the class (intent) of the input sentence."""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    """Fetches a response based on the predicted intent."""
    if not intents_list:
        return "Sorry, I couldn't understand that. Could you please rephrase?"
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def listen_for_audio():
    """Captures and transcribes audio input from the user."""
    with sr.Microphone() as source:
        print("| Listening... (speak your query)")
        try:
            audio = recognizer.listen(source)
            print("| Recognizing...")
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("| Sorry, I couldn't understand that.")
            return ""
        except sr.RequestError:
            print("| Sorry, there was an issue with the speech recognition service.")
            return ""

def speak(response_text):
    """Converts chatbot's response text to speech using pyttsx3."""
    engine = pyttsx3.init()
    engine.say(response_text)
    engine.runAndWait()

# Main chatbot loop
print("|=================== Welcome to Clinic Chatbot! ======================|")
print("|============================== Feel Free ============================|")
print("|================================== To ===============================|")
print("|=============== Ask any query about our Clinic =====================|")

# Ask user to choose input method
while True:
    print("| Choose an input method: (1) Text, (2) Voice")
    choice = input("| Your Choice: ")
    if choice in ["1", "2"]:
        break
    print("| Invalid choice. Please select 1 or 2.")

while True:
    if choice == "1":
        message = input("| You (text): ")
    elif choice == "2":
        message = listen_for_audio()

    if not message:
        continue

    if message.lower() in ["bye", "goodbye"]:
        print("|===================== The Program Ends Here! =====================|")
        break

    # Predict the intent and generate a response
    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    print("| Your Assistant:", response)

    # Speak the response using pyttsx3
    speak(response)

