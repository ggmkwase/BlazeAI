import gradio as gr
from blaze_core import blaze_chat  # Your existing chat function
import time

# Custom CSS for a premium look
custom_css = """
:root {
    --primary: #6e6ef8;
    --secondary: #2b2b2b;
    --accent: #ff4d4d;
}

.gradio-container {
    background: var(--secondary) !important;
    color: white !important;
    font-family: 'Segoe UI', sans-serif !important;
}

.dark .chatbot {
    background: #1e1e1e !important;
    border-radius: 12px !important;
}

.dark .chatbot .message.user {
    background: #2a2a2a !important;
    border-left: 3px solid var(--primary) !important;
}

.dark .chatbot .message.bot {
    background: #252525 !important;
    border-left: 3px solid var(--accent) !important;
}

#restart-btn {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
}

#header {
    text-align: center;
    padding: 1em;
    background: linear-gradient(90deg, var(--primary), #8a2be2);
    color: white;
    border-radius: 12px;
    margin-bottom: 1em;
}
"""

def add_timestamp():
    return f"🕒 {time.strftime('%H:%M:%S')}"

with gr.Blocks(theme=gr.themes.Default(primary_hue="purple"), css=custom_css) as demo:
    # Header
    gr.Markdown("""
    <div id="header">
        <h1>BlazeAI</h1>
        <p>Next-Gen AI Assistant</p>
    </div>
    """)

    # Main Chat Interface
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat History",
                bubble_full_width=False,
                height=600,
                avatar_images=(
                    "https://i.imgur.com/7TSaSlF.png",  # User avatar
                    "https://i.imgur.com/7TSaSlF.png"   # Bot avatar
                )
            )
            msg = gr.Textbox(
                placeholder="Type your message...",
                lines=2,
                max_lines=5,
                label="Message"
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear History")
                restart_btn = gr.Button("♻️ Restart", elem_id="restart-btn")

        # Sidebar with controls
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            model = gr.Dropdown(
                ["llama3-70b", "mixtral-8x7b", "GPT-4"],
                label="AI Model",
                value="llama3-70b"
            )
            temperature = gr.Slider(0, 1, 0.7, label="Creativity")
            timestamp = gr.Textbox(add_timestamp(), label="Last Updated")

    # Event Handlers
    def respond(message, chat_history, model, temp):
        bot_message = blaze_chat(message)  # Your existing function
        chat_history.append((message, bot_message))
        timestamp = add_timestamp()
        return "", chat_history, timestamp

    msg.submit(
        respond,
        [msg, chatbot, model, temperature],
        [msg, chatbot, timestamp]
    )
    submit_btn.click(
        respond,
        [msg, chatbot, model, temperature],
        [msg, chatbot, timestamp]
    )
    clear_btn.click(lambda: None, None, chatbot)
    restart_btn.click(lambda: time.sleep(3), None, timestamp)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        favicon_path="https://i.imgur.com/7TSaSlF.png"
    )