from dotenv import load_dotenv
load_dotenv()

# utils.py
import os
import csv
from datetime import datetime

LOG_FILE = os.getenv("LOG_FILE")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

def apply_mistral_chat_template(messages):
    conversation = "<s>"
    conversation += f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"

    first_user = True
    for msg in messages:
        role = msg.get("role") or msg.get("from")  # soporta 'role' o 'from'
        content = msg.get("content") or msg.get("value")
        
        if role in ["user", "human", "USER", "HUMAN"]:
            if first_user:
                # El primer usuario va directo después del system prompt, sin [INST]
                conversation += f"{content} [/INST] "
                first_user = False
            else:
                # Los siguientes mensajes de usuario abren [INST]
                conversation += f"[INST] {content} [/INST] "
        elif role in ["assistant", "ASSISTANT"]:
            # respuesta de la IA justo después del [/INST] del usuario
            conversation += f"{content} "

    conversation = conversation.strip()
    return conversation


def log_response(prompt, temperature, response):
    """Guarda la respuesta en CSV para tracking/testing."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "prompt", "temperature", "response"])
        writer.writerow([datetime.now().isoformat(), prompt, temperature, response])