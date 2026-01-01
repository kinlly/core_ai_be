from dotenv import load_dotenv
load_dotenv()

import torch
import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from pathlib import Path
from file_utils import load_jsonl, save_jsonl, load_txt

from utils import apply_mistral_chat_template, log_response
from auth import login_huggingface

login_huggingface()

MAIN_DATA_DIR = Path(os.getenv("MAIN_DATA_DIR"))
BOOK_DATA_PATH = Path(os.getenv("BOOK_DATA_PATH"))
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_ADAPTER_DIR = os.getenv("MODEL_ADAPTER_DIR")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
LOG_FILE = os.getenv("LOG_FILE")
SEED = os.getenv("SEED")
ADAPTER_DIR = MODEL_ADAPTER_DIR
OFFLOAD_DIR = os.path.join(MODEL_ADAPTER_DIR, "offload_infer")

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
    logging.info(f"Archivos en {MODEL_ADAPTER_DIR}: {os.listdir(MODEL_ADAPTER_DIR)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ADAPTER_DIR, trust_remote_code=True)
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

class ConversationRequest(BaseModel):
    conversation: list[dict]  # [{"role": "user"/"assistant", "content": "..."}]
    temperatures: list[float] = [0, 0.3, 0.5, 0.7]
    max_new_tokens: int = 128
    seed: int = SEED

@app.post("/generate")
async def generate_response(request: ConversationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")
    
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + request.conversation
    formatted_prompt = apply_mistral_chat_template(msgs)
    
    print(formatted_prompt)   
    
    candidates = []

    for temp in request.temperatures:
        try:
            if SEED != "-1":
                logging.info(f"Adding Seed {request.seed}")
                torch.manual_seed(request.seed)
            else:
                logging.info(f"Skipping Seed")

            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                if temp == 0:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=request.max_new_tokens,
                        do_sample=False,
                        repetition_penalty=1.1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                else:
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

            assistant_response = text.split("[/INST]")[-1].strip().split("\n")[0]
            
            log_response(formatted_prompt, temp, assistant_response)

            candidates.append({"temperature": temp, "response": assistant_response})
            logging.info(f"[Temp {temp}] {assistant_response}")

        except Exception as e:
            logging.error(f"⚠️ Error generando respuesta a temp {temp}: {e}")
            candidates.append({"temperature": temp, "response": ""})

    candidates.append({"temperature": "manual", "response": ""})
    
    return {"candidates": candidates}

@app.get("/files")
def list_files():
    if not MAIN_DATA_DIR.exists():
        raise HTTPException(404, detail="Directorio no encontrado")

    files = [
        {
            "name": f.name,
            "size": f.stat().st_size  # tamaño en bytes
        }
        for f in MAIN_DATA_DIR.glob("*.jsonl")
    ]
    return {"files": files}

@app.get("/files/{filename}")
def read_file(filename: str):
    path = MAIN_DATA_DIR / filename
    if not path.exists():
        raise HTTPException(404, detail="Fichero no encontrado")
    return load_jsonl(path)

@app.post("/files/{filename}")
def add_line(filename: str, record: dict):
    path = MAIN_DATA_DIR / filename
    if not path.exists():
        raise HTTPException(404, detail="Fichero no encontrado")
    records = load_jsonl(path)
    records.append(record)
    save_jsonl(path, records)
    return {"status": "ok", "total": len(records)}

@app.put("/files/{filename}/{index}")
def update_line(filename: str, index: int, record: dict):
    path = MAIN_DATA_DIR / filename
    if not path.exists():
        raise HTTPException(404, detail="Fichero no encontrado")
    records = load_jsonl(path)
    if index < 0 or index >= len(records):
        raise HTTPException(400, detail="Índice fuera de rango")
    records[index] = record
    save_jsonl(path, records)
    return {"status": "ok", "updated_index": index}

@app.delete("/files/{filename}/{index}")
def delete_line(filename: str, index: int):
    path = MAIN_DATA_DIR / filename
    if not path.exists():
        raise HTTPException(404, detail="Fichero no encontrado")
    records = load_jsonl(path)
    if index < 0 or index >= len(records):
        raise HTTPException(400, detail="Índice fuera de rango")
    removed = records.pop(index)
    save_jsonl(path, records)
    return {"status": "ok", "deleted": removed}

@app.get("/book")
def get_all_lines():
    """Devuelve todas las líneas del archivo BOOK_DATA_PATH"""
    path = BOOK_DATA_PATH
    if not path.exists():
        raise HTTPException(404, detail="Archivo no encontrado")
    return load_txt(path)

@app.get("/book/{chapter}")
def get_line(chapter: int):
    """Devuelve una línea específica por índice"""
    path = BOOK_DATA_PATH
    if not path.exists():
        raise HTTPException(404, detail="Archivo no encontrado")
    lines = load_txt(path)
    if chapter < 0 or chapter >= len(lines):
        raise HTTPException(400, detail="Índice fuera de rango")
    return {"index": chapter-1, "line": lines[chapter-1]}

class LineUpdate(BaseModel):
    line: str

@app.put("/book/{index}")
def update_line(index: int, payload: LineUpdate):
    path = BOOK_DATA_PATH
    if not path.exists():
        raise HTTPException(404, detail="Archivo no encontrado")
    lines = load_txt(path)
    if index < 0 or index >= len(lines):
        raise HTTPException(400, detail="Índice fuera de rango")
    lines[index] = payload.line
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {"status": "ok", "updated_index": index}

TARGET_DIR = Path(r"C:\repos\PenguinProto\datafiles")
TARGET_DIR_SCENES = TARGET_DIR  / "scenes"
TARGET_DIR_EQUIP = TARGET_DIR  / "equip"
TARGET_DIR_ENEMIES = TARGET_DIR  / "enemies"
TARGET_DIR_CHARACTERS = TARGET_DIR  / "characters"

@app.put("/editor/json/{filename}")
def update_json(filename: str, record: dict):
    
    if not filename.lower().endswith(".json"):
        filename += ".json"
    
    path: Path = TARGET_DIR_SCENES / filename
    
    try:
        TARGET_DIR_SCENES.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Error al acceder al directorio {TARGET_DIR_SCENES}: {e}")
        raise HTTPException(status_code=500, detail="Error al acceder al directorio de destino.")

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
        
        logging.info(f"🔄 JSON actualizado/sobrescrito correctamente en: {path}")
        return {"status": "ok", "filename": filename, "path": str(path)}
    except Exception as e:
        logging.error(f"⚠️ Error al actualizar el archivo JSON en {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al actualizar el archivo: {e}")

@app.get("/editor/json")
def list_json_files():
    if not TARGET_DIR_SCENES.exists():
        raise HTTPException(404, detail=f"Directorio de destino no encontrado: {TARGET_DIR_SCENES}")

    files_data = []

    for f in TARGET_DIR_SCENES.glob("*.json"):
        if f.is_file():
            file_info = {
                "name": f.name,
                "size": f.stat().st_size,  # tamaño en bytes
                "path": str(f.resolve()), # ruta absoluta
                "content": None # Inicializamos el contenido
            }
            
            try:
                with f.open("r", encoding="utf-8") as file:
                    file_info["content"] = json.load(file)
                
            except json.JSONDecodeError:
                logging.warning(f"⚠️ Error de formato JSON en el archivo: {f.name}. Se devuelve 'null' en 'content'.")
            except Exception as e:
                logging.error(f"⚠️ Error al leer el archivo {f.name}: {e}")
            
            files_data.append(file_info)

    return {"files": files_data}

@app.post("/editor/json/{filename}")
def add_json(filename: str, record: dict):
    if not filename.lower().endswith(".json"):
        filename += ".json"
    
    path: Path = TARGET_DIR_SCENES / filename
    
    try:
        TARGET_DIR_SCENES.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Error al crear el directorio {TARGET_DIR_SCENES}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al acceder al directorio de destino.")

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
        
        logging.info(f"✅ JSON guardado correctamente en la ruta fija: {path}")
        return {"status": "ok", "filename": filename, "path": str(path)}
    except Exception as e:
        logging.error(f"⚠️ Error al guardar el archivo JSON en {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {e}")

def load_json():
    EQUIP_PATH = TARGET_DIR_EQUIP / "equip.json"
    if not EQUIP_PATH.exists():
        return {}
    try:
        with EQUIP_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_json(data):
    EQUIP_PATH = TARGET_DIR_EQUIP / "equip.json"
    TARGET_DIR_EQUIP.mkdir(parents=True, exist_ok=True)
    with EQUIP_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# --- PATH & UTILS ---

EQUIP_PATH = TARGET_DIR_EQUIP / "equip.json"
# RootData ahora es un diccionario plano {item_id: item_data}
RootData = dict[str, dict]

def load_json() -> RootData:
    if not os.path.exists(EQUIP_PATH):
        # Inicializamos con la estructura correcta vacía (un diccionario vacío)
        return {}
    try:
        with open(EQUIP_PATH, "r") as f:
            data = json.load(f)
            # Nos aseguramos de que sea un diccionario, si no, devolvemos uno vacío
            return data if isinstance(data, dict) else {}
    except:
        return {}

def save_json(data: RootData):
    with open(EQUIP_PATH, "w") as f:
        # Usamos indent=4 para mantenerlo legible
        json.dump(data, f, indent=4)

# --- ENDPOINTS (Actualizados) ---

@app.get("/editor/equip")
def get_equip():
    """Retorna la lista completa de equip plana."""
    # Retorna directamente el diccionario plano de ítems.
    return load_json()

@app.post("/editor/equip/{item_name}")
def create_item(item_name: str, item_data: dict):
    """Crea un item con una clave de ID única."""
    data = load_json()
    
    # Verificamos si la clave ID ya existe en la lista plana
    if item_name in data:
        raise HTTPException(status_code=400, detail=f"El item ID '{item_name}' ya existe.")
    
    data[item_name] = item_data
    save_json(data)
    # Ya no necesitamos retornar la 'category'
    return {"status": "created", "item": item_name}

@app.put("/editor/equip/{item_name}")
def update_item(item_name: str, item_data: dict):
    """Actualiza un item específico en tiempo real por su ID."""
    data = load_json()
    
    if item_name not in data:
        raise HTTPException(status_code=404, detail="Item no encontrado")
        
    data[item_name] = item_data
    save_json(data)
    return {"status": "updated", "item": item_name}

@app.delete("/editor/equip/{item_name}")
def delete_item(item_name: str):
    """Elimina un item específico por su ID."""
    data = load_json()
    
    if item_name not in data:
        raise HTTPException(status_code=404, detail="Item no encontrado")
        
    del data[item_name]
    save_json(data)
    return {"status": "deleted", "item": item_name}


TARGET_DIR_RESOURCES = TARGET_DIR / "resources" 
RESOURCES_FILE = TARGET_DIR_RESOURCES / "resources.json"

def load_resources():
    # Aseguramos que la carpeta exista
    TARGET_DIR_RESOURCES.mkdir(parents=True, exist_ok=True)
    
    if not RESOURCES_FILE.exists():
        with open(RESOURCES_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}
        
    try:
        with open(RESOURCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logging.error(f"Error cargando recursos: {e}")
        return {}

def save_resources(data):
    # Aseguramos que la carpeta exista antes de guardar
    TARGET_DIR_RESOURCES.mkdir(parents=True, exist_ok=True)
    with open(RESOURCES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# --- ENDPOINTS ---
@app.get("/editor/resources")
def get_resources():
    """Retorna la lista completa de recursos."""
    return load_resources()

@app.post("/editor/resources/{res_id}")
def create_resource(res_id: str, res_data: dict):
    """Crea un recurso (ej: res_id='wood_01')."""
    data = load_resources()
    if res_id in data:
        raise HTTPException(status_code=400, detail=f"El recurso '{res_id}' ya existe.")
    
    # Estructura mínima sugerida: { "name": "", "description": "", "type": "" }
    data[res_id] = res_data
    save_resources(data)
    return {"status": "created", "id": res_id}

@app.put("/editor/resources/{res_id}")
def update_resource(res_id: str, res_data: dict):
    """Actualiza un recurso existente."""
    data = load_resources()
    if res_id not in data:
        raise HTTPException(status_code=404, detail="Recurso no encontrado")
        
    data[res_id] = res_data
    save_resources(data)
    return {"status": "updated", "id": res_id}

@app.delete("/editor/resources/{res_id}")
def delete_resource(res_id: str):
    """Elimina un recurso."""
    data = load_resources()
    if res_id not in data:
        raise HTTPException(status_code=404, detail="Recurso no encontrado")
        
    del data[res_id]
    save_resources(data)
    return {"status": "deleted", "id": res_id}

# --- PATH PARA MISIONES ---
TARGET_DIR_QUESTS = TARGET_DIR / "quests"
QUESTS_FILE = TARGET_DIR_QUESTS / "quests.json"

def load_quests():
    TARGET_DIR_QUESTS.mkdir(parents=True, exist_ok=True)
    if not QUESTS_FILE.exists():
        with open(QUESTS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}
    with open(QUESTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_quests(data):
    with open(QUESTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# --- ENDPOINTS MISIONES ---

@app.get("/editor/quests")
def get_quests():
    return load_quests()

@app.get("/editor/catalog/all-items")
def get_all_possible_items():
    """Endpoint de conveniencia que combina Equip y Resources."""
    equip = load_json() # Tu función de equipos
    resources = load_resources() # Tu función de recursos
    
    # Marcamos el origen para que el FE sepa qué es qué
    combined = {}
    for k, v in equip.items():
        combined[k] = { **v, "origin": "equip" }
    for k, v in resources.items():
        combined[k] = { **v, "origin": "resource" }
    
    return combined

@app.post("/editor/quests/{quest_id}")
def create_quest(quest_id: str, quest_data: dict):
    data = load_quests()
    if quest_id in data:
        raise HTTPException(status_code=400, detail="ID de misión ya existe")
    data[quest_id] = quest_data
    save_quests(data)
    return {"status": "created", "id": quest_id}

@app.put("/editor/quests/{quest_id}")
def update_quest(quest_id: str, quest_data: dict):
    data = load_quests()
    data[quest_id] = quest_data
    save_quests(data)
    return {"status": "updated", "id": quest_id}

ENEMIES_PATH = TARGET_DIR_ENEMIES / "enemies.json"
EnemyRootData = dict[str, dict]

def load_enemies_json() -> EnemyRootData:
    """Carga los datos de enemigos desde enemies.json."""
    if not os.path.exists(ENEMIES_PATH):
        return {}
    try:
        with open(ENEMIES_PATH, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except:
        return {}

def save_enemies_json(data: EnemyRootData):
    """Guarda los datos de enemigos en enemies.json."""
    with open(ENEMIES_PATH, "w") as f:
        # Usamos indent=4 para mantenerlo legible
        json.dump(data, f, indent=4)

# --- ENDPOINTS CRUD DE ENEMIGOS ---

@app.get("/editor/enemies")
def get_enemies():
    """Retorna el diccionario completo y plano de enemigos."""
    return load_enemies_json()

@app.post("/editor/enemies/{enemy_id}")
def create_enemy(enemy_id: str, enemy_data: dict):
    """Crea un nuevo enemigo con una clave de ID única."""
    data = load_enemies_json()
    
    if enemy_id in data:
        raise HTTPException(status_code=400, detail=f"El enemigo ID '{enemy_id}' ya existe.")
    
    data[enemy_id] = enemy_data
    save_enemies_json(data)
    return {"status": "created", "enemy_id": enemy_id}

@app.put("/editor/enemies/{enemy_id}")
def update_enemy(enemy_id: str, enemy_data: dict):
    """Actualiza un enemigo específico por su ID."""
    data = load_enemies_json()
    
    if enemy_id not in data:
        raise HTTPException(status_code=404, detail="Enemigo no encontrado")
        
    data[enemy_id] = enemy_data
    save_enemies_json(data)
    return {"status": "updated", "enemy_id": enemy_id}

@app.delete("/editor/enemies/{enemy_id}")
def delete_enemy(enemy_id: str):
    """Elimina un enemigo específico por su ID."""
    data = load_enemies_json()
    
    if enemy_id not in data:
        raise HTTPException(status_code=404, detail="Enemigo no encontrado")
        
    del data[enemy_id]
    save_enemies_json(data)
    return {"status": "deleted", "enemy_id": enemy_id}

TARGET_DIR_CHARACTERS = TARGET_DIR  / "characters"
CHARACTERS_PATH = TARGET_DIR_CHARACTERS / "characters.json"
CharacterRootData = dict[str, dict]

def load_characters_json() -> CharacterRootData:
    """Carga los datos de personajes desde characters.json."""
    if not os.path.exists(CHARACTERS_PATH):
        return {}
    try:
        with open(CHARACTERS_PATH, "r") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except:
        return {}
def save_characters_json(data: CharacterRootData):
    """Guarda los datos de personajes en characters.json."""
    with open(CHARACTERS_PATH, "w") as f:
        # Usamos indent=4 para mantenerlo legible
        json.dump(data, f, indent=4)
# --- ENDPOINTS CRUD DE PERSONAJES ---
@app.get("/editor/characters")
def get_characters():
    """Retorna el diccionario completo y plano de personajes."""
    return load_characters_json()
@app.post("/editor/characters/{character_id}")
def create_character(character_id: str, character_data: dict):
    """Crea un nuevo personaje con una clave de ID única."""
    data = load_characters_json()
    
    if character_id in data:
        raise HTTPException(status_code=400, detail=f"El personaje ID '{character_id}' ya existe.")
    
    data[character_id] = character_data
    save_characters_json(data)
    return {"status": "created", "character_id": character_id}
@app.put("/editor/characters/{character_id}")
def update_character(character_id: str, character_data: dict):
    """Actualiza un personaje específico por su ID."""
    data = load_characters_json()
    
    if character_id not in data:
        raise HTTPException(status_code=404, detail="Personaje no encontrado")
        
    data[character_id] = character_data
    save_characters_json(data)
    return {"status": "updated", "character_id": character_id}
@app.delete("/editor/characters/{character_id}")
def delete_character(character_id: str):
    """Elimina un personaje específico por su ID."""
    data = load_characters_json()
    
    if character_id not in data:
        raise HTTPException(status_code=404, detail="Personaje no encontrado")
        
    del data[character_id]
    save_characters_json(data)
    return {"status": "deleted", "character_id": character_id}
