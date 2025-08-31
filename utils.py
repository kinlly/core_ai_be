# utils.py
def apply_mistral_chat_template(messages, add_generation_prompt=True):
    conversation = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            conversation += f"[INST]{content} [/INST]\n"
        elif role == "user":
            conversation += f"[INST]{content} [/INST]\n"
        elif role == "assistant":
            conversation += f"{content}\n"
    if add_generation_prompt:
        conversation += "\n"
    return conversation

def log_response(prompt, temperature, response):
    """Guarda la respuesta en CSV para tracking/testing."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "prompt", "temperature", "response"])
        writer.writerow([datetime.now().isoformat(), prompt, temperature, response])