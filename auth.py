# auth.py
import os
from huggingface_hub import login

def login_huggingface():
    """Carga la key desde .env y hace login en Hugging Face Hub."""
    
    huggingface_key = os.getenv("HUGGINGFACE_KEY")
    
    if not huggingface_key:
        raise ValueError("❌ No se encontró HUGGINGFACE_KEY en el .env")
    
    login(token=huggingface_key)
    print("✅ Login en HuggingFace correcto")