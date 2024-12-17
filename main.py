import random
import json
import pickle
import numpy as np
import nltk
import speech_recognition as sr
import pyttsx3

from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
recognizer = sr.Recognizer()

# Load trained chatbot model and data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Appointment management
appointments = {}  # {phone_number: {"name": "User Name", "slot": "10:00 AM"}}
available_slots = ["9:00 AM", "10:00 AM", "11:00 AM", "12:00 PM", "2:00 PM", "3:00 PM", "4:00 PM"]

# Helper Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]


def get_response(intent, intents_json):
    tag = intent[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses']), tag
    return "I'm sorry, I didn't understand that.", None


def listen_for_audio():
    """Capture and transcribe audio input from the user."""
    with sr.Microphone() as source:
        print("Listening... Please speak your query.")
        try:
            audio = recognizer.listen(source)
            print("Recognizing...")
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Could you repeat?")
            return ""
        except sr.RequestError:
            print("Sorry, there was an issue with the speech recognition service.")
            return ""


def speak(response_text):
    """Convert the chatbot's response text to speech."""
    engine = pyttsx3.init()
    engine.say(response_text)
    engine.runAndWait()


def show_slots():
    if available_slots:
        return f"Here are the available slots: {', '.join(available_slots)}"
    return "No slots are currently available. Please try again later."


def book_slot(phone, name, slot):
    if phone in appointments:
        return f"Sorry, you have already booked an appointment at {appointments[phone]['slot']}."
    if slot not in available_slots:
        return f"Sorry, {slot} is not available. Please choose another slot."
    appointments[phone] = {"name": name, "slot": slot}
    available_slots.remove(slot)
    return f"Appointment booked successfully for {name} at {slot}."


def cancel_appointment(phone):
    if phone in appointments:
        appointment = appointments.pop(phone)
        available_slots.append(appointment["slot"])
        available_slots.sort()  # Keep slots ordered
        return f"Your appointment at {appointment['slot']} has been canceled successfully."
    return "No appointment found for the provided phone number."


# Main Chatbot Logic
def chatbot():
    user_context = {}
    print("Welcome to the Appointment Booking Assistant!")
    print("You can ask about our services, book appointments, or cancel them.")
    print("Choose an input method: (1) Text, (2) Voice")
    choice = input("Your choice: ").strip()

    while True:
        if choice == "1":
            user_input = input("You: ").strip()
        elif choice == "2":
            user_input = listen_for_audio()

        if not user_input:
            continue

        if user_input.lower() in ['exit', 'bye']:
            print("Assistant: Goodbye! Have a great day!")
            speak("Goodbye! Have a great day!")
            break

        intents_list = predict_class(user_input)
        response, tag = get_response(intents_list, intents)

        # Handle appointment booking flow
        if tag == "book_appointment":
            if "name" not in user_context:
                response = "May I know your name, please?"
                user_context["context"] = "get_name"
            elif "phone" not in user_context:
                response = "Can I have your phone number for the booking?"
                user_context["context"] = "get_phone"
            elif "slot" not in user_context:
                response = show_slots() + " Which time slot would you like to book?"
                user_context["context"] = "get_slot"
            else:
                name = user_context["name"]
                phone = user_context["phone"]
                slot = user_context["slot"]
                response = book_slot(phone, name, slot)
                user_context.clear()

        elif tag == "cancel_appointment":
            response = "Can you provide the phone number used for the booking?"
            user_context["context"] = "cancel_phone"

        elif user_context.get("context") == "cancel_phone":
            if user_input.isdigit():
                response = cancel_appointment(user_input)
                user_context.clear()
            else:
                response = "Please enter a valid phone number."

        elif user_context.get("context") == "get_name":
            user_context["name"] = user_input
            response = "Thank you! Now, may I have your phone number?"
            user_context["context"] = "get_phone"

        elif user_context.get("context") == "get_phone":
            if user_input.isdigit():
                user_context["phone"] = user_input
                response = show_slots() + " Which time slot would you like to book?"
                user_context["context"] = "get_slot"
            else:
                response = "Please enter a valid phone number."

        elif user_context.get("context") == "get_slot":
            user_context["slot"] = user_input
            name = user_context["name"]
            phone = user_context["phone"]
            response = book_slot(phone, name, user_input)
            user_context.clear()

        # Print and speak the response
        print("Assistant:", response)
        speak(response)


if __name__ == "__main__":
    chatbot()
