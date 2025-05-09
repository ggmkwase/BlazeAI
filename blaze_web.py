# blaze_web.py
import gradio as gr
from blaze_core import generate_response
from updater import update_blaze

def web_response(chat_history, username, prompt, personality):
    response = generate_response(username, prompt, personality)
    chat_history.append((prompt, response))
    return chat_history, ""

def clear_chat():
    return [], ""

def start_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🔥 <span style='color:#ff4d4d;'>Blaze AI</span> - Personal Assistant", elem_id="title")

        with gr.Row():
            username = gr.Textbox(label="🧑 Username", placeholder="Enter your name")
            personality = gr.Dropdown(["default", "chill", "tech"], value="default", label="🧠 Personality")

        chatbot = gr.Chatbot(label="🗨️ Conversation", height=400)
        prompt = gr.Textbox(label="💬 Your Message", placeholder="Type your message or command...", lines=1)

        with gr.Row():
            send_btn = gr.Button("➡️ Send")
            clear_btn = gr.Button("🧹 Clear Chat")
            update_btn = gr.Button("🔄 Update Blaze")

        send_btn.click(fn=web_response, inputs=[chatbot, username, prompt, personality],
                       outputs=[chatbot, prompt])
        clear_btn.click(fn=clear_chat, outputs=[chatbot, prompt])
        update_btn.click(fn=lambda: update_blaze(), outputs=[])

    demo.launch()

