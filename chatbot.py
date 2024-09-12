import streamlit as st
import speech_recognition as sr
import os
import nltk
import cv2
import tempfile
import google.generativeai as genai
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from moviepy.editor import VideoFileClip
import pyttsx3  # Import pyttsx3
from dotenv import load_dotenv

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load environment variables
load_dotenv()

# Configure Gemini Pro API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

def recognize_speech_from_audio_file(file_path):
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        st.error(f"Error recognizing speech from the audio file: {e}")
        return None

def speech_to_text(audio_data):
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Sorry, I did not understand that. Try again recording."
    except sr.RequestError:
        return "Sorry, there seems to be a network issue."

def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

def generate_response(text):
    response = get_gemini_response(text)
    
    final_response = ""
    try:
        for chunk in response:
            if hasattr(chunk, 'text'):
                final_response += chunk.text + " "
            else:
                st.error("Unexpected response format.")
                return ""
    except Exception as e:
        st.error(f"Error processing response: {e}")
        return ""
                    
    return final_response.strip()
recording_flag = True
def record_video():
    global recording_flag
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return None

    st.info("Recording video...")

    while recording_flag:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the resulting frame
        cv2.imshow('Live Video', frame)

        # Press 'q' to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            recording_flag = False

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()
    st.success("Video recording stopped.")
    
    # Save the recorded video
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_path = temp_video_file.name
    # Save the video (code to save video needs to be added)
    
    return temp_video_path

def extract_audio_from_video(video_file_path):
    clip = VideoFileClip(video_file_path)
    if clip.audio is None:
        st.error("No audio track found in the video.")
        return None

    audio_path = video_file_path.replace(".mp4", "_audio.wav")
    clip.audio.write_audiofile(audio_path)
    return audio_path

def speak_text(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")

# Streamlit Interface
st.title("Speech-to-Speech Chatbot with Microphone, Audio/video File, and Video Input")

input_option = st.selectbox("Choose input method:", ('Microphone', 'Upload Audio/video File', 'Record Video'))

if input_option == 'Microphone':
    user_input = ""
    response = ""
    recording = False

    status_placeholder = st.empty()

    if st.button("Record from Microphone"):
        recording = True
        status_placeholder.write("Listening...")

        with sr.Microphone() as source:
            audio = recognizer.listen(source)
            transcription = speech_to_text(audio)
        status_placeholder.empty()
        recording = False
        if transcription:
            st.write(f"You said: {transcription}")
            st.info("Generating response...")
            response = generate_response(transcription)
            st.write(f"Generated Response: {response}")
            speak_text(response)

elif input_option == 'Upload Audio/video File':
    uploaded_file = st.file_uploader("Upload an audio file (.mp3, .wav, .mp4)", type=["mp3", "wav", "mp4"])

    if uploaded_file:
        os.makedirs("temp", exist_ok=True)

        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".mp4"):
            audio_path = extract_audio_from_video(file_path)
        else:
            audio_path = file_path

        if audio_path:
            transcription = recognize_speech_from_audio_file(audio_path)
            if transcription:
                st.write(f"Transcribed Text: {transcription}")
                st.info("Generating response...")
                response = generate_response(transcription)
                st.write(f"Generated Response: {response}")
                speak_text(response)

elif input_option == 'Record Video':
    if st.button("Start Recording"):
        video_path = record_video()
        if video_path:
            audio_path = extract_audio_from_video(video_path)
            if audio_path:
                transcription = recognize_speech_from_audio_file(audio_path)
                if transcription:
                    st.write(f"Transcribed Text: {transcription}")
                    st.info("Generating response...")
                    response = generate_response(transcription)
                    st.write(f"Generated Response: {response}")
                    speak_text(response)
