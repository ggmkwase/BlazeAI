# blaze_core.py
import random
import json
import os
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
import wikipedia
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a model (e.g., GPT-2, or any Hugging Face model)
model_name = "gpt2"  # Replace with your preferred model (e.g., "facebook/blenderbot-400M-distill")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure GPU usage if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

MEMORY_DIR = "memory"

if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)
    
# Load or create user profiles
def load_user_profile(username):
    try:
        with open("user_profiles.json", "r") as f:
            profiles = json.load(f)
        return profiles.get(username, {"personality": "default"})
    except FileNotFoundError:
        return {"personality": "default"}

def save_user_profile(username, personality):
    try:
        with open("user_profiles.json", "r") as f:
            profiles = json.load(f)
    except FileNotFoundError:
        profiles = {}
    profiles[username] = {"personality": personality}
    with open("user_profiles.json", "w") as f:
        json.dump(profiles, f, indent=4)

def update_personality(username, personality):
    save_user_profile(username, personality)
    

# Initialize sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

tasks = []

def add_task(task_description):
    tasks.append({"task": task_description, "status": "Pending"})
    return f"Task added: {task_description}"

def list_tasks():
    return "\n".join([f"{i+1}. {task['task']} - {task['status']}" for i, task in enumerate(tasks)])

def mark_task_complete(task_number):
    tasks[task_number-1]['status'] = "Completed"
    return f"Task {task_number} marked as completed!"


def analyze_sentiment(text):
    sentiment = sentiment_model(text)
    return sentiment[0]['label'], sentiment[0]['score']


# Initialize speech-to-text and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening for your command...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            return "Sorry, I did not understand that."



# Personalities
personalities = {
    "default": "I am Blaze, your helpful assistant.",
    "chill": "Hey! I'm Blaze, just vibing here to help you.",
    "tech": "Blaze AI ready. Systems optimal and listening.",
    "funny": "I'm Blaze, your sarcastic sidekick!"
}

def get_user_memory(username):
    filepath = os.path.join(MEMORY_DIR, f"{username}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {"history": [], "tasks": []}

def save_user_memory(username, memory):
    filepath = os.path.join(MEMORY_DIR, f"{username}.json")
    with open(filepath, "w") as f:
        json.dump(memory, f)

def generate_response(username, prompt, personality="default"):
    memory = get_user_memory(username)
    memory["history"].append(prompt)
    memory["history"] = memory["history"][-20:]  # Keep memory small
    save_user_memory(username, memory)

    intro = personalities.get(personality, personalities["default"])
    response = f"{intro}\nYou said: {prompt}\nMemory entries: {len(memory['history'])}"
    return response

def add_task(username, task):
    memory = get_user_memory(username)
    memory["tasks"].append({"task": task, "done": False})
    save_user_memory(username, memory)
    return f"Task added: {task}"

def show_tasks(username):
    memory = get_user_memory(username)
    if not memory["tasks"]:
        return "No tasks yet."
    return "\n".join([
        f"[{i+1}] {'✅' if t['done'] else '❌'} {t['task']}" 
        for i, t in enumerate(memory["tasks"])
    ])

def complete_task(username, index):
    memory = get_user_memory(username)
    try:
        memory["tasks"][index]["done"] = True
        save_user_memory(username, memory)
        return f"Marked task #{index + 1} as done."
    except IndexError:
        return "Invalid task number."

def clear_tasks(username):
    memory = get_user_memory(username)
    memory["tasks"] = []
    save_user_memory(username, memory)
    return "All tasks cleared."



def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Sorry, there are multiple results. Could you clarify? ({str(e.options)})"

    chat_history = {}

def get_chat_history(username):
    return chat_history.get(username, [])

def add_to_history(username, prompt, response):
    if username not in chat_history:
        chat_history[username] = []
    chat_history[username].append((prompt, response))

# Main function for Blaze interaction
def blaze_chat(username, message):
    # Load user profile and detect sentiment
    user_profile = load_user_profile(username)
    sentiment, score = analyze_sentiment(message)

    # Add to chat history
    add_to_history(username, message, "Response here...")
    
    # Task management feature (example)
    if "add task" in message:
        task = message.replace("add task", "").strip()
        add_task(task)
        return f"Task added: {task}"

    # Example sentiment-based response
    if sentiment == "NEGATIVE":
        return f"Sorry to hear that. Here's something positive: Keep going, you're doing great!"
    else:
        return f"Glad to hear you're in a good mood! How can I assist you further?"

# Create the Gradio interface
iface = gr.Interface(fn=blaze_chat, 
                     inputs=[gr.Textbox(label="Username"), gr.Textbox(label="Message")],
                     outputs="text",
                     live=True)

# Launch the Gradio interface
iface.launch()