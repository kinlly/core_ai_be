import torch
import os
import logging
import csv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
huggingface_key = os.getenv("HUGGINGFACE_KEY")
login(token=huggingface_key)

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
FINAL_DIR = r"C:\repos\mistral-finetune"
ADAPTER_DIR = FINAL_DIR
OFFLOAD_DIR = os.path.join(FINAL_DIR, "offload_infer")

SYSTEM_PROMPT = """Eres Yui. Responde SIEMPRE en primera persona ("yo"), con tono directo y conciso.
No uses etiquetas como "Yui:" ni narrador en tercera persona.
No repitas la instrucción del usuario.
Siempre responde en español (castellano).
Canon (no contradigas):
- Género Mujer.
"""

LOG_FILE = "response_log.csv"
SEED = 3407

app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None

# -------------------------
# MODELO
# -------------------------
try:
    logging.info(f"Archivos en {FINAL_DIR}: {os.listdir(FINAL_DIR)}")

    tokenizer = AutoTokenizer.from_pretrained(FINAL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        offload_folder=OFFLOAD_DIR,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=False)
    model.to(device).eval()
    logging.info(f"✅ Modelo cargado correctamente en {device}.")

except Exception as e:
    logging.error(f"Error cargando modelo: {e}")
    model = None
    tokenizer = None

# -------------------------
# UTIL
# -------------------------
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

# -------------------------
# API
# -------------------------
class ConversationRequest(BaseModel):
    conversation: list[dict]  # [{"role": "user"/"assistant", "content": "..."}]
    temperatures: list[float] = [0.3, 0.5, 0.7]
    max_new_tokens: int = 128
    seed: int = SEED

@app.post("/generate")
async def generate_response(request: ConversationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")
    
    # Construir prompt con historial
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + request.conversation
    formatted_prompt = apply_mistral_chat_template(msgs, add_generation_prompt=True)
    
    candidates = []

    for temp in request.temperatures:
        try:
            # Fijar semilla por temperatura
            torch.manual_seed(request.seed)

            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            # Extraer respuesta tras [/INST]
            assistant_response = text.split("[/INST]")[-1].strip().split("\n")[0]
            
            # Guardar log
            log_response(formatted_prompt, temp, assistant_response)

            # Añadir al listado de candidatos
            candidates.append({"temperature": temp, "response": assistant_response})
            logging.info(f"[Temp {temp}] {assistant_response}")

        except Exception as e:
            logging.error(f"⚠️ Error generando respuesta a temp {temp}: {e}")
            candidates.append({"temperature": temp, "response": ""})

    # Añadir empty para input manual
    candidates.append({"temperature": "manual", "response": ""})
    
    return {"candidates": candidates}
