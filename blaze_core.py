# blaze_core.py
import random
import json
import os

MEMORY_DIR = "memory"

if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

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

