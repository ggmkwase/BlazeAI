import gradio as gr
import os
import sys
from threading import Thread

# Import your existing chatbot logic
from blaze_core import blaze_chat

def restart_app():
    """Restarts the application"""
    os.execl(sys.executable, sys.executable, *sys.argv)

def safe_restart():
    """Triggers restart in a background thread"""
    Thread(target=restart_app).start()
    return "🔄 Restarting BlazeAI... Refresh your browser in 5 seconds."

# Gradio interface with restart button
with gr.Blocks() as demo:
    with gr.Tab("Chat"):
        gr.ChatInterface(blaze_chat)
    
    with gr.Tab("Developer"):
        gr.Markdown("### Server Controls")
        restart_btn = gr.Button("♻️ Restart Server", variant="stop")
        status = gr.Textbox(label="Status")
        restart_btn.click(safe_restart, outputs=status)

if __name__ == "__main__":
    demo.launch()