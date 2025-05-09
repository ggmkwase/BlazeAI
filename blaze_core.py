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
import sqlite3
from github import Github
from dotenv import load_dotenv
from github import Github, InputGitAuthor
from git import Repo
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# GitHub Setup
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_PATH = os.path.dirname(os.path.abspath(__file__))  # Assumes script is in repo root
REPO_NAME = "your_username/your_repo"  # Change this

class GitHubAgent:
    def __init__(self):
        self.g = Github(GITHUB_TOKEN)
        self.repo = self.g.get_repo(REPO_NAME)
        self.local_repo = Repo(REPO_PATH)
        self.author = InputGitAuthor(
            name="BlazeAI-Bot",
            email="blazeai@example.com"
        )

    def analyze_file(self, filepath):
        """Analyze a file and return potential issues"""
        try:
            content = self.repo.get_contents(filepath).decoded_content.decode()
            
            # AI analysis logic (simplified example)
            issues = []
            if "try:" not in content and "except:" not in content:
                issues.append("Missing error handling in critical sections")
            if "json.load" in content and not any(x in content for x in ["JSONDecodeError", "Exception"]):
                issues.append("Unsafe JSON parsing without error handling")
            
            return {
                "file": filepath,
                "issues": issues,
                "content_sample": content[:200] + "..." if len(content) > 200 else content
            }
        except Exception as e:
            return {"error": str(e)}

    def create_issue(self, title, body, labels=["bug"]):
        """Create a GitHub issue"""
        return self.repo.create_issue(
            title=title,
            body=body,
            labels=labels
        )

    def create_pr(self, branch_name, changes, pr_title, pr_body):
        """Create a pull request with fixes"""
        # Create new branch
        self.local_repo.git.checkout('HEAD', b=branch_name)
        
        # Apply changes (simplified - in reality you'd modify files)
        for filepath, new_content in changes.items():
            with open(os.path.join(REPO_PATH, filepath), 'w') as f:
                f.write(new_content)
        
        # Commit and push
        self.local_repo.git.add(A=True)
        self.local_repo.index.commit(f"BlazeAI auto-fix: {pr_title}", author=self.author)
        self.local_repo.git.push('origin', branch_name)
        
        # Create PR
        return self.repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=branch_name,
            base="main"
        )

def blaze_chat(username, message):
    github_agent = GitHubAgent()
    
    if "analyze" in message.lower():
        # Extract filename from message like "analyze blaze_core.py"
        filepath = message.lower().replace("analyze", "").strip()
        analysis = github_agent.analyze_file(filepath)
        
        if "error" in analysis:
            return f"Analysis failed: {analysis['error']}"
        
        response = f"🔍 Analysis of {filepath}:\n"
        response += f"📄 Sample: {analysis['content_sample']}\n\n"
        
        if analysis['issues']:
            response += "⚠️ Found potential issues:\n- " + "\n- ".join(analysis['issues'])
            
            # Auto-create issue for serious problems
            if "Unsafe JSON parsing" in "\n".join(analysis['issues']):
                issue = github_agent.create_issue(
                    title=f"Security: Unsafe JSON parsing in {filepath}",
                    body=f"BlazeAI detected unhandled JSON parsing in {filepath}\n\n```python\n{analysis['content_sample']}\n```"
                )
                response += f"\n\n🚨 Created GitHub issue: #{issue.number}"
        else:
            response += "✅ No critical issues found"
        
        return response
    
    elif "fix" in message.lower():
        # Example auto-fix workflow
        branch_name = f"blazeai-fix-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        changes = {
            "blaze_core.py": "# Your fixed code here\n...",
        }
        
        pr = github_agent.create_pr(
            branch_name=branch_name,
            changes=changes,
            pr_title="[BlazeAI] Auto-fix for JSON parsing",
            pr_body="Automated fix for unsafe JSON parsing detected by BlazeAI"
        )
        return f"🔧 Created PR #{pr.number}: {pr.title}\n{pr.html_url}"
    
    # ... (rest of your existing chat logic)
    return "I can help analyze/fix this repo! Try:\n- 'analyze blaze_core.py'\n- 'fix JSON parsing'"

# Load a model (e.g., GPT-2, or any Hugging Face model)
model_name = "gpt2"  # Replace with your preferred model (e.g., "facebook/blenderbot-400M-distill")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure GPU usage if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

load_dotenv()  # Load environment variables

def connect_to_github():
    try:
        g = Github(os.getenv("GITHUB_TOKEN"))
        repo = g.get_repo("your_username/your_repo")  # e.g., "ggmkwase/blazeAI"
        return repo
    except Exception as e:
        print(f"GitHub connection failed: {e}")
        return None

def analyze_code(repo, filepath):
    """Let AI review a specific file"""
    try:
        file_content = repo.get_contents(filepath).decoded_content.decode()
        # Add your AI analysis logic here
        return f"Analyzed {filepath}:\n{file_content[:500]}..."  # Truncated for demo
    except Exception as e:
        return f"Error analyzing {filepath}: {e}"

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (username TEXT, timestamp TEXT, user_msg TEXT, bot_msg TEXT)''')
    conn.commit()
    conn.close()

MEMORY_DIR = "memory"

if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)
    
# Load or create user profiles
def init_chat_history(username):
    if not os.path.exists(f"chat_history_{username}.json"):
        with open(f"chat_history_{username}.json", "w") as f:
            json.dump([], f)

def load_user_profile(username):
    try:
        with open("user_profiles.json", "r") as f:
            profiles = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        profiles = {}  # Fallback to empty dict
    
    return profiles.get(username, {})

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
    try:
        with open(f"chat_history_{username}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print(f"Error reading chat history for {username}, resetting...")
        return []

def add_to_history(username, user_message, bot_message):
    history = get_chat_history(username)  # Changed from chat_history to get_chat_history()
    history.append({"user": user_message, "bot": bot_message})
    with open(f"chat_history_{username}.json", "w") as f:
        json.dump(history, f)

# Main function for Blaze interaction
from groq import Groq

def blaze_chat(username, message):
   
    if "analyze repo" in message.lower():
        repo = connect_to_github()
        if repo:
            return analyze_code(repo, "blaze_core.py")  # Example: Analyze core file
        else:
            return "Failed to connect to GitHub"
   # Initialize Groq client
    client = Groq(api_key="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")
    
    # Generate AI response
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": message}],
        model="llama3-70b-8192",  # Free model
    )
    response = chat_completion.choices[0].message.content
    
    # Save to history
    add_to_history(username, message, response)
    return response

# Create the Gradio interface
iface = gr.Interface(fn=blaze_chat, 
                     inputs=[gr.Textbox(label="Username"), gr.Textbox(label="Message")],
                     outputs="text",
                     live=True)

# Launch the Gradio interface
iface.launch()