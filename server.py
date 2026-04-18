from dotenv import load_dotenv
load_dotenv()

import torch
import os
import re
import json
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime, timezone

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from pathlib import Path
import shutil

EXTRA_BACKUP_DIR = Path(r"C:\repos\ylbtm\metadata")

def backup_file(path: Path):
    try:
        EXTRA_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup_path = EXTRA_BACKUP_DIR / path.name
            shutil.copy2(path, backup_path)
            logging.info(f"Backup creado en {backup_path}")
    except Exception as e:
        logging.error(f"Error creando backup: {e}")
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
    backup_file(path)
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
    backup_file(path)
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
    backup_file(path)
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
    backup_file(path)
    return {"status": "ok", "updated_index": index}

TARGET_DIR = Path(r"C:\repos\ylbtm\metadata")
TARGET_DIR_SCENES = TARGET_DIR / "scenes"
TARGET_DIR_TIMELINES = TARGET_DIR / "timelines"
TRANSLATIONS_DIR = TARGET_DIR / "translations"
ALL_TIMELINE_FILENAME = "all_timeline.json"
ALL_TIMELINE_TITLE = "all timeline"
TIMELINE_DEFAULT_X = 80
TIMELINE_DEFAULT_Y = 120
TIMELINE_HORIZONTAL_GAP = 520
TIMELINE_LAYOUTS_DIR = Path(__file__).resolve().parent / "editor_state" / "timelines"
ALL_TIMELINE_LAYOUT_PATH = TIMELINE_LAYOUTS_DIR / "all_timeline_layout.json"
CHAPTER_TIMELINE_RE = re.compile(r"^chapter[_-]?(\d+)\.json$", re.IGNORECASE)
FULL_VIEW_ALIAS_FILENAMES = {"full-view.json", "full_view.json"}

def _normalize_editor_filename(filename: str, label: str) -> str:
    safe_name = Path(filename).name.strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail=f"Nombre de {label} inválido.")
    if not safe_name.lower().endswith(".json"):
        safe_name += ".json"
    return safe_name

def _normalize_scene_filename(filename: str) -> str:
    return _normalize_editor_filename(filename, "escena")

def _normalize_timeline_filename(filename: str) -> str:
    return _normalize_editor_filename(filename, "timeline")

def _timeline_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _is_all_timeline_filename(filename: str) -> bool:
    return filename.lower() == ALL_TIMELINE_FILENAME

def _is_full_view_alias(filename: str) -> bool:
    return filename.lower() in FULL_VIEW_ALIAS_FILENAMES

def _timeline_name_sort_key(filename: str):
    match = CHAPTER_TIMELINE_RE.match(filename)
    if match:
        return (0, int(match.group(1)), filename.lower())
    return (1, filename.lower())

def _timeline_card_sort_key(card: dict, fallback_index: int):
    position = card.get("position", {}) if isinstance(card, dict) else {}
    if isinstance(position, dict):
        x = position.get("x")
        y = position.get("y")
    else:
        x = None
        y = None

    fallback = _default_all_timeline_position(fallback_index)
    card_x = x if isinstance(x, (int, float)) else fallback["x"]
    card_y = y if isinstance(y, (int, float)) else fallback["y"]
    return (card_x, card_y, fallback_index)

def _default_all_timeline_position(index: int) -> dict:
    return {
        "x": TIMELINE_DEFAULT_X + index * TIMELINE_HORIZONTAL_GAP,
        "y": TIMELINE_DEFAULT_Y + (index % 2) * 36,
    }

def _timeline_filename_to_chapter_label(filename: str) -> str:
    match = CHAPTER_TIMELINE_RE.match(filename)
    if match:
        return f"chapter_{match.group(1).zfill(2)}"
    return Path(filename).stem

def _load_all_timeline_layout() -> tuple[dict, str | None]:
    if not ALL_TIMELINE_LAYOUT_PATH.exists():
        return {}, None

    try:
        with ALL_TIMELINE_LAYOUT_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError:
        logging.warning(f"⚠️ Error de formato JSON en el layout del all timeline: {ALL_TIMELINE_LAYOUT_PATH}")
        return {}, None
    except Exception as e:
        logging.error(f"⚠️ Error al leer el layout del all timeline {ALL_TIMELINE_LAYOUT_PATH}: {e}")
        return {}, None

    if not isinstance(payload, dict):
        return {}, None

    positions = payload.get("positions", {})
    if not isinstance(positions, dict):
        return {}, None

    normalized_positions: dict[str, dict[str, int]] = {}
    for card_id, raw_position in positions.items():
        if not isinstance(card_id, str) or not isinstance(raw_position, dict):
            continue
        x = raw_position.get("x")
        y = raw_position.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        normalized_positions[card_id] = {
            "x": round(float(x)),
            "y": round(float(y)),
        }

    updated_at = payload.get("updatedAt")
    return normalized_positions, updated_at if isinstance(updated_at, str) else None

def _save_all_timeline_layout(record: dict) -> dict:
    cards = record.get("cards", []) if isinstance(record, dict) else []
    if not isinstance(cards, list):
        raise HTTPException(status_code=400, detail="El all timeline necesita una lista válida de cards.")

    positions: dict[str, dict] = {}
    for raw_card in cards:
        if not isinstance(raw_card, dict):
            continue
        card_id = raw_card.get("id")
        position = raw_card.get("position")
        if not isinstance(card_id, str) or not card_id.strip() or not isinstance(position, dict):
            continue
        x = position.get("x")
        y = position.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue

        entry = {
            "x": round(float(x)),
            "y": round(float(y)),
        }
        source_timeline_name = raw_card.get("sourceTimelineName")
        chapter_label = raw_card.get("chapterLabel")
        if isinstance(source_timeline_name, str) and source_timeline_name.strip():
            entry["sourceTimelineName"] = source_timeline_name.strip()
        if isinstance(chapter_label, str) and chapter_label.strip():
            entry["chapterLabel"] = chapter_label.strip()
        positions[card_id.strip()] = entry

    payload = {
        "version": 1,
        "title": ALL_TIMELINE_TITLE,
        "updatedAt": _timeline_now_iso(),
        "positions": positions,
    }

    try:
        TIMELINE_LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
        with ALL_TIMELINE_LAYOUT_PATH.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"⚠️ Error al guardar el layout del all timeline en {ALL_TIMELINE_LAYOUT_PATH}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar el layout del all timeline: {e}")

    return payload

def _write_timeline_document(path: Path, record: dict):
    with path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=4, ensure_ascii=False)
    backup_file(path)

def _list_real_timeline_files() -> list[dict]:
    if not TARGET_DIR_TIMELINES.exists():
        return []

    files_data: list[dict] = []

    for f in sorted(TARGET_DIR_TIMELINES.glob("*.json"), key=lambda path: _timeline_name_sort_key(path.name)):
        if not f.is_file():
            continue
        if _is_all_timeline_filename(f.name):
            continue

        file_info = {
            "name": f.name,
            "size": f.stat().st_size,
            "path": str(f.resolve()),
            "content": None,
        }

        try:
            with f.open("r", encoding="utf-8") as file:
                file_info["content"] = json.load(file)
        except json.JSONDecodeError:
            logging.warning(f"⚠️ Error de formato JSON en el timeline: {f.name}. Se devuelve 'null' en 'content'.")
        except Exception as e:
            logging.error(f"⚠️ Error al leer el timeline {f.name}: {e}")

        files_data.append(file_info)

    return files_data

def _build_all_timeline_file(source_files: list[dict]) -> dict | None:
    if len(source_files) == 0:
        return None

    layout_positions, layout_updated_at = _load_all_timeline_layout()
    aggregated_cards: list[dict] = []
    latest_updated_at = layout_updated_at

    for source_file in sorted(source_files, key=lambda item: _timeline_name_sort_key(str(item.get("name", "")))):
        filename = source_file.get("name")
        content = source_file.get("content")
        if not isinstance(filename, str) or not isinstance(content, dict):
            continue

        raw_cards = content.get("cards", [])
        if not isinstance(raw_cards, list):
            continue

        chapter_label = _timeline_filename_to_chapter_label(filename)
        sorted_cards = sorted(
            (card for card in raw_cards if isinstance(card, dict)),
            key=lambda card: _timeline_card_sort_key(card, raw_cards.index(card)),
        )

        for card in sorted_cards:
            card_id = card.get("id")
            if not isinstance(card_id, str) or not card_id.strip():
                continue

            default_position = _default_all_timeline_position(len(aggregated_cards))
            position = layout_positions.get(card_id, default_position)
            updated_at = card.get("updatedAt")
            if isinstance(updated_at, str) and (latest_updated_at is None or updated_at > latest_updated_at):
                latest_updated_at = updated_at

            aggregated_cards.append({
                "id": card_id,
                "markdown": card.get("markdown", "") if isinstance(card.get("markdown"), str) else "",
                "position": {
                    "x": position.get("x", default_position["x"]),
                    "y": position.get("y", default_position["y"]),
                },
                "color": card.get("color", "#ee9b64") if isinstance(card.get("color"), str) else "#ee9b64",
                "updatedAt": updated_at if isinstance(updated_at, str) else _timeline_now_iso(),
                "sourceTimelineName": filename,
                "chapterLabel": chapter_label,
            })

    content = {
        "version": 1,
        "title": ALL_TIMELINE_TITLE,
        "updatedAt": latest_updated_at or _timeline_now_iso(),
        "cards": aggregated_cards,
    }

    return {
        "name": ALL_TIMELINE_FILENAME,
        "displayName": "Full View",
        "size": len(json.dumps(content, ensure_ascii=False)),
        "path": str(ALL_TIMELINE_LAYOUT_PATH.resolve()),
        "isAggregate": True,
        "isReadonlyContent": False,
        "sourceCount": len(source_files),
        "content": content,
    }

def _build_empty_full_view_file() -> dict:
    content = {
        "version": 1,
        "title": "full view",
        "updatedAt": _timeline_now_iso(),
        "cards": [],
    }

    return {
        "name": ALL_TIMELINE_FILENAME,
        "displayName": "Full View",
        "size": len(json.dumps(content, ensure_ascii=False)),
        "path": str(ALL_TIMELINE_LAYOUT_PATH.resolve()),
        "isAggregate": True,
        "isReadonlyContent": False,
        "sourceCount": 0,
        "content": content,
    }

def _sync_full_view(record: dict) -> dict:
    if not isinstance(record, dict):
        raise HTTPException(status_code=400, detail="El full view necesita un documento válido.")

    raw_cards = record.get("cards", [])
    if not isinstance(raw_cards, list):
        raise HTTPException(status_code=400, detail="El full view necesita una lista válida de cards.")

    source_files = _list_real_timeline_files()
    source_documents: dict[str, dict] = {}
    card_sources_by_id: dict[str, str] = {}

    for source_file in source_files:
        filename = source_file.get("name")
        content = source_file.get("content")
        path_str = source_file.get("path")
        if not isinstance(filename, str) or not isinstance(content, dict) or not isinstance(path_str, str):
            continue

        source_documents[filename] = {
            "path": Path(path_str),
            "content": content,
        }

        source_cards = content.get("cards", [])
        if not isinstance(source_cards, list):
            continue

        for source_card in source_cards:
            if not isinstance(source_card, dict):
                continue
            card_id = source_card.get("id")
            if isinstance(card_id, str) and card_id.strip():
                card_sources_by_id[card_id.strip()] = filename

    payload_cards_by_id: dict[str, dict] = {}
    for raw_card in raw_cards:
        if not isinstance(raw_card, dict):
            raise HTTPException(status_code=400, detail="Hay cards inválidas en el full view.")

        card_id = raw_card.get("id")
        if not isinstance(card_id, str) or not card_id.strip():
            raise HTTPException(status_code=400, detail="Todas las cards del full view necesitan un ID válido.")
        card_id = card_id.strip()

        expected_source = card_sources_by_id.get(card_id)
        if expected_source is None:
            raise HTTPException(status_code=400, detail=f"La card '{card_id}' no existe en los timelines origen.")

        source_timeline_name = raw_card.get("sourceTimelineName")
        if isinstance(source_timeline_name, str) and source_timeline_name.strip():
            source_timeline_name = source_timeline_name.strip()
        else:
            source_timeline_name = expected_source

        if source_timeline_name != expected_source:
            raise HTTPException(status_code=400, detail=f"La card '{card_id}' pertenece a '{expected_source}', no a '{source_timeline_name}'.")

        if card_id in payload_cards_by_id:
            raise HTTPException(status_code=400, detail=f"La card '{card_id}' aparece duplicada en el full view.")

        payload_card = dict(raw_card)
        payload_card["sourceTimelineName"] = expected_source
        payload_cards_by_id[card_id] = payload_card

    modified_timelines: list[str] = []
    removed_cards = 0
    updated_cards = 0

    for timeline_name, source_entry in source_documents.items():
        content = source_entry["content"]
        source_cards = content.get("cards", [])
        if not isinstance(source_cards, list):
            continue

        next_cards: list = []
        timeline_changed = False

        for source_card in source_cards:
            if not isinstance(source_card, dict):
                next_cards.append(source_card)
                continue

            card_id = source_card.get("id")
            if not isinstance(card_id, str) or not card_id.strip():
                next_cards.append(source_card)
                continue
            card_id = card_id.strip()

            payload_card = payload_cards_by_id.get(card_id)
            if payload_card is None:
                removed_cards += 1
                timeline_changed = True
                continue

            next_card = dict(source_card)
            card_changed = False

            payload_markdown = payload_card.get("markdown")
            if isinstance(payload_markdown, str) and payload_markdown != source_card.get("markdown"):
                next_card["markdown"] = payload_markdown
                card_changed = True

            payload_color = payload_card.get("color")
            if isinstance(payload_color, str) and payload_color.strip() and payload_color != source_card.get("color"):
                next_card["color"] = payload_color
                card_changed = True

            if card_changed:
                payload_updated_at = payload_card.get("updatedAt")
                next_card["updatedAt"] = payload_updated_at if isinstance(payload_updated_at, str) and payload_updated_at.strip() else _timeline_now_iso()
                updated_cards += 1
                timeline_changed = True

            next_cards.append(next_card)

        if timeline_changed:
            content["cards"] = next_cards
            content["updatedAt"] = _timeline_now_iso()
            _write_timeline_document(source_entry["path"], content)
            modified_timelines.append(timeline_name)

    layout_payload = _save_all_timeline_layout({
        "cards": list(payload_cards_by_id.values()),
    })

    return {
        "status": "ok",
        "path": str(ALL_TIMELINE_LAYOUT_PATH.resolve()),
        "saved_cards": len(layout_payload.get("positions", {})),
        "modified_timelines": modified_timelines,
        "removed_cards": removed_cards,
        "updated_cards": updated_cards,
    }

# ── Translation helpers ──────────────────────────────────────────────────────
LANG_CODES = ["es", "en", "pt-br", "pl", "zh-cn", "es-419", "de", "ja", "fr", "ru", "ko", "tr", "it"]

def _load_translation(category: str) -> dict:
    path = TRANSLATIONS_DIR / f"{category}.translation.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_translation(category: str, data: dict):
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSLATIONS_DIR / f"{category}.translation.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def _ensure_translation_entry(category: str, key: str, item: dict, field_mapping: dict):
    """Create translation entry for a newly created item (ES seeded, others empty)."""
    data = _load_translation(category)
    if key in data:
        return
    data[key] = {}
    for trans_field, src_field in field_mapping.items():
        es_value = item.get(src_field, "")
        data[key][trans_field] = {lang: (es_value if lang == "es" else "") for lang in LANG_CODES}
    _save_translation(category, data)

def _sync_es_translation(category: str, key: str, old_item: dict, new_item: dict, field_mapping: dict):
    """If a translatable field changed: update ES, clear all other languages."""
    data = _load_translation(category)
    if key not in data:
        data[key] = {}
    changed = False
    for trans_field, src_field in field_mapping.items():
        new_val = new_item.get(src_field, "")
        old_val = old_item.get(src_field, "")
        if new_val != old_val:
            data[key][trans_field] = {lang: (new_val if lang == "es" else "") for lang in LANG_CODES}
            changed = True
    if changed:
        _save_translation(category, data)

def _delete_translation_entry(category: str, key: str):
    """Remove translation entry when its source item is deleted."""
    data = _load_translation(category)
    if key in data:
        del data[key]
        _save_translation(category, data)
# ─────────────────────────────────────────────────────────────────────────────

@app.put("/editor/json/{filename}")
def update_json(filename: str, record: dict):
    filename = _normalize_scene_filename(filename)
    
    path: Path = TARGET_DIR_SCENES / filename
    
    try:
        TARGET_DIR_SCENES.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Error al acceder al directorio {TARGET_DIR_SCENES}: {e}")
        raise HTTPException(status_code=500, detail="Error al acceder al directorio de destino.")

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
        backup_file(path)
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
    filename = _normalize_scene_filename(filename)
    
    path: Path = TARGET_DIR_SCENES / filename
    
    try:
        TARGET_DIR_SCENES.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Error al crear el directorio {TARGET_DIR_SCENES}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al acceder al directorio de destino.")

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
        backup_file(path)
        logging.info(f"✅ JSON guardado correctamente en la ruta fija: {path}")
        return {"status": "ok", "filename": filename, "path": str(path)}
    except Exception as e:
        logging.error(f"⚠️ Error al guardar el archivo JSON en {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {e}")

@app.post("/editor/json/{filename}/rename")
def rename_json(filename: str, new_filename: str):
    current_filename = _normalize_scene_filename(filename)
    target_filename = _normalize_scene_filename(new_filename)

    if current_filename == target_filename:
        current_path = TARGET_DIR_SCENES / current_filename
        return {"status": "ok", "filename": current_filename, "path": str(current_path.resolve())}

    current_path = TARGET_DIR_SCENES / current_filename
    target_path = TARGET_DIR_SCENES / target_filename

    if not current_path.exists():
        raise HTTPException(status_code=404, detail=f"La escena '{current_filename}' no existe.")
    if target_path.exists():
        raise HTTPException(status_code=400, detail=f"La escena '{target_filename}' ya existe.")

    try:
        TARGET_DIR_SCENES.mkdir(parents=True, exist_ok=True)
        backup_file(current_path)
        current_path.rename(target_path)
        backup_file(target_path)
        logging.info(f"📝 Escena renombrada: {current_filename} -> {target_filename}")
        return {"status": "ok", "filename": target_filename, "path": str(target_path.resolve())}
    except Exception as e:
        logging.error(f"⚠️ Error al renombrar la escena {current_filename} -> {target_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al renombrar la escena: {e}")

@app.put("/editor/timelines/{filename}")
def update_timeline(filename: str, record: dict):
    filename = _normalize_timeline_filename(filename)

    if _is_full_view_alias(filename):
        logging.info("🔄 Sincronizando full view mediante ruta dedicada.")
        return _sync_full_view(record)

    if _is_all_timeline_filename(filename):
        logging.info("🔄 Sincronizando full view mediante alias de all timeline.")
        return _sync_full_view(record)

    path: Path = TARGET_DIR_TIMELINES / filename

    try:
        TARGET_DIR_TIMELINES.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Error al acceder al directorio {TARGET_DIR_TIMELINES}: {e}")
        raise HTTPException(status_code=500, detail="Error al acceder al directorio de timelines.")

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
        backup_file(path)
        logging.info(f"🔄 Timeline actualizado/sobrescrito correctamente en: {path}")
        return {"status": "ok", "filename": filename, "path": str(path)}
    except Exception as e:
        logging.error(f"⚠️ Error al actualizar el timeline en {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al actualizar el timeline: {e}")

@app.get("/editor/timelines")
def list_timeline_files():
    return {"files": _list_real_timeline_files()}

@app.get("/editor/timelines/full-view")
def get_full_view_timeline():
    source_files = _list_real_timeline_files()
    full_view_file = _build_all_timeline_file(source_files)
    if full_view_file is None:
        full_view_file = _build_empty_full_view_file()
    return {"file": full_view_file}

@app.put("/editor/timelines/full-view")
def update_full_view_timeline(record: dict):
    return _sync_full_view(record)

@app.post("/editor/timelines/{filename}")
def add_timeline(filename: str, record: dict):
    filename = _normalize_timeline_filename(filename)

    if _is_all_timeline_filename(filename) or _is_full_view_alias(filename):
        raise HTTPException(status_code=400, detail="El all timeline se genera automáticamente y no se puede crear manualmente.")

    path: Path = TARGET_DIR_TIMELINES / filename

    try:
        TARGET_DIR_TIMELINES.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Error al crear el directorio {TARGET_DIR_TIMELINES}: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al acceder al directorio de timelines.")

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
        backup_file(path)
        logging.info(f"✅ Timeline guardado correctamente en: {path}")
        return {"status": "ok", "filename": filename, "path": str(path)}
    except Exception as e:
        logging.error(f"⚠️ Error al guardar el timeline en {path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar el timeline: {e}")

@app.post("/editor/timelines/{filename}/rename")
def rename_timeline(filename: str, new_filename: str):
    current_filename = _normalize_timeline_filename(filename)
    target_filename = _normalize_timeline_filename(new_filename)

    if _is_all_timeline_filename(current_filename) or _is_all_timeline_filename(target_filename) or _is_full_view_alias(current_filename) or _is_full_view_alias(target_filename):
        raise HTTPException(status_code=400, detail="El all timeline es virtual y no se puede renombrar.")

    if current_filename == target_filename:
        current_path = TARGET_DIR_TIMELINES / current_filename
        return {"status": "ok", "filename": current_filename, "path": str(current_path.resolve())}

    current_path = TARGET_DIR_TIMELINES / current_filename
    target_path = TARGET_DIR_TIMELINES / target_filename

    if not current_path.exists():
        raise HTTPException(status_code=404, detail=f"El timeline '{current_filename}' no existe.")
    if target_path.exists():
        raise HTTPException(status_code=400, detail=f"El timeline '{target_filename}' ya existe.")

    try:
        TARGET_DIR_TIMELINES.mkdir(parents=True, exist_ok=True)
        backup_file(current_path)
        current_path.rename(target_path)
        backup_file(target_path)
        logging.info(f"📝 Timeline renombrado: {current_filename} -> {target_filename}")
        return {"status": "ok", "filename": target_filename, "path": str(target_path.resolve())}
    except Exception as e:
        logging.error(f"⚠️ Error al renombrar el timeline {current_filename} -> {target_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error al renombrar el timeline: {e}")

# --- PATH & UTILS ---

EQUIP_PATH = TARGET_DIR / "equip.json"
# RootData ahora es un diccionario plano {item_id: item_data}
RootData = dict[str, dict]

def load_json() -> RootData:
    if not os.path.exists(EQUIP_PATH):
        return {}
    try:
        with open(EQUIP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_json(data: RootData):
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    with open(EQUIP_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    backup_file(EQUIP_PATH)


# --- ENDPOINTS (Actualizados) ---

@app.get("/editor/equip")
def get_equip():
    """Retorna la lista completa de equip plana."""
    # Retorna directamente el diccionario plano de ítems.
    return load_json()

_EQUIP_TRANS = {"displayDescription": "description"}

@app.post("/editor/equip/{item_name}")
def create_item(item_name: str, item_data: dict):
    """Crea un item con una clave de ID única."""
    data = load_json()
    if item_name in data:
        raise HTTPException(status_code=400, detail=f"El item ID '{item_name}' ya existe.")
    data[item_name] = item_data
    save_json(data)
    _ensure_translation_entry("equip", item_name, item_data, _EQUIP_TRANS)
    return {"status": "created", "item": item_name}

@app.put("/editor/equip/{item_name}")
def update_item(item_name: str, item_data: dict):
    """Actualiza un item específico en tiempo real por su ID."""
    data = load_json()
    if item_name not in data:
        raise HTTPException(status_code=404, detail="Item no encontrado")
    old = data[item_name]
    data[item_name] = item_data
    save_json(data)
    _sync_es_translation("equip", item_name, old, item_data, _EQUIP_TRANS)
    return {"status": "updated", "item": item_name}

@app.delete("/editor/equip/{item_name}")
def delete_item(item_name: str):
    """Elimina un item específico por su ID."""
    data = load_json()
    if item_name not in data:
        raise HTTPException(status_code=404, detail="Item no encontrado")
    del data[item_name]
    save_json(data)
    _delete_translation_entry("equip", item_name)
    return {"status": "deleted", "item": item_name}


SPRITES_DIR = Path(r"C:\repos\ylbtm\assets\sprites")

@app.get("/editor/sprite/{sprite_name}")
def get_sprite(sprite_name: str):
    """Serves a sprite PNG from the ylbtm sprites directory.
    Accepts Godot sprite keys (e.g. spr_sword_T01) or plain filenames (e.g. sword_001).
    Mapping: spr_type_Tnn → type_nnn.png
    """
    safe_name = Path(sprite_name).name  # prevent path traversal

    candidates = [safe_name]

    if safe_name.startswith("spr_"):
        stripped = safe_name[4:]  # remove "spr_"
        # spr_sword_T01 → sword_001
        normalized = re.sub(r"_T(\d+)$", lambda m: f"_{int(m.group(1)):03d}", stripped)
        candidates.append(normalized)
        # fallback to base type sprite (e.g. sword.png)
        base = stripped.split("_")[0]
        candidates.append(base)

    for name in candidates:
        path = SPRITES_DIR / f"{name}.png"
        if path.resolve().parent == SPRITES_DIR.resolve() and path.exists():
            return FileResponse(path, media_type="image/png")

    raise HTTPException(status_code=404, detail="Sprite not found")


RESOURCES_FILE = TARGET_DIR / "resources.json"

def load_resources():
    if not RESOURCES_FILE.exists():
        return {}
    try:
        with open(RESOURCES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logging.error(f"Error cargando recursos: {e}")
        return {}

def save_resources(data):
    with open(RESOURCES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    backup_file(RESOURCES_FILE)

_RESOURCES_TRANS = {"displayName": "name", "displayDescription": "description"}

# --- ENDPOINTS ---
@app.get("/editor/resources")
def get_resources():
    """Retorna la lista completa de recursos."""
    return load_resources()

@app.post("/editor/resources/{res_id}")
def create_resource(res_id: str, res_data: dict):
    data = load_resources()
    if res_id in data:
        raise HTTPException(status_code=400, detail=f"El recurso '{res_id}' ya existe.")
    data[res_id] = res_data
    save_resources(data)
    _ensure_translation_entry("resources", res_id, res_data, _RESOURCES_TRANS)
    return {"status": "created", "id": res_id}

@app.put("/editor/resources/{res_id}")
def update_resource(res_id: str, res_data: dict):
    data = load_resources()
    if res_id not in data:
        raise HTTPException(status_code=404, detail="Recurso no encontrado")
    old = data[res_id]
    data[res_id] = res_data
    save_resources(data)
    _sync_es_translation("resources", res_id, old, res_data, _RESOURCES_TRANS)
    return {"status": "updated", "id": res_id}

@app.delete("/editor/resources/{res_id}")
def delete_resource(res_id: str):
    data = load_resources()
    if res_id not in data:
        raise HTTPException(status_code=404, detail="Recurso no encontrado")
    del data[res_id]
    save_resources(data)
    _delete_translation_entry("resources", res_id)
    return {"status": "deleted", "id": res_id}

# --- PATH PARA MISIONES ---
QUESTS_FILE = TARGET_DIR / "quests.json"

def load_quests():
    if not QUESTS_FILE.exists():
        return {}
    with open(QUESTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_quests(data):
    with open(QUESTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    backup_file(QUESTS_FILE)

_QUESTS_TRANS = {"displayName": "title", "displayDescription": "description"}

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
    _ensure_translation_entry("quests", quest_id, quest_data, _QUESTS_TRANS)
    return {"status": "created", "id": quest_id}

@app.put("/editor/quests/{quest_id}")
def update_quest(quest_id: str, quest_data: dict):
    data = load_quests()
    old = data.get(quest_id, {})
    data[quest_id] = quest_data
    save_quests(data)
    _sync_es_translation("quests", quest_id, old, quest_data, _QUESTS_TRANS)
    return {"status": "updated", "id": quest_id}

@app.delete("/editor/quests/{quest_id}")
def delete_quest(quest_id: str):
    data = load_quests()
    if quest_id not in data:
        raise HTTPException(status_code=404, detail="Misión no encontrada")
    del data[quest_id]
    save_quests(data)
    _delete_translation_entry("quests", quest_id)
    return {"status": "deleted", "id": quest_id}

ENEMIES_PATH = TARGET_DIR / "enemies.json"
EnemyRootData = dict[str, dict]

def load_enemies_json() -> EnemyRootData:
    """Carga los datos de enemigos desde enemies.json."""
    if not os.path.exists(ENEMIES_PATH):
        return {}
    try:
        with open(ENEMIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except:
        return {}

def save_enemies_json(data: EnemyRootData):
    """Guarda los datos de enemigos en enemies.json."""
    with open(ENEMIES_PATH, "w", encoding="utf-8") as f:
        # Usamos indent=4 para mantenerlo legible
        json.dump(data, f, indent=4, ensure_ascii=False)
    backup_file(ENEMIES_PATH)

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

CHARACTERS_PATH = TARGET_DIR / "characters.json"
CharacterRootData = dict[str, dict]
CHARACTER_DESIGN_TYPES = (
    "Shinigami Attack",
    "Shinigami Magic",
    "Shinigami 0",
    "Shinigami Defense",
    "Lazo",
    "DiosElemental",
    "Demon",
    "Desconocido",
)
LEGACY_CHARACTER_TYPE_MAP = {
    "shinigami_attack": "Shinigami Attack",
    "shinigami_magic": "Shinigami Magic",
    "shinigami_0": "Shinigami 0",
    "shinigami_def": "Shinigami Defense",
    "lazo": "Lazo",
    "nagas": "Desconocido",
    "omeg": "Desconocido",
    "rey": "Desconocido",
    "civil": "Desconocido",
    "teacher": "Desconocido",
}
SKIN_TONES = ("very_pale", "pale", "light", "tan", "dark", "deep_dark")
BODY_TYPES = ("petite", "slim", "athletic", "defined", "muscular", "bulky", "tall_slender")
HEIGHT_CATEGORIES = ("very_short", "short", "average", "tall", "very_tall")
HAIR_LENGTHS = ("bald", "very_short", "short", "medium", "long", "very_long")
HAIR_TEXTURES = ("straight", "wavy", "curly", "coily", "spiky", "messy")
HAIR_STYLES = ("loose", "ponytail", "side_ponytail", "twintails", "braided", "half_up", "bun", "topknot", "tied_back", "layered", "short_cut", "undercut", "ceremonial", "custom")
EYE_SPECIALS = ("none", "glow", "heterochromia", "dual_color", "slit_pupil", "void_pupil", "cross_pattern", "ring_pattern", "covered")
EYE_INTENSITIES = ("low", "medium", "high", "extreme")
FACE_STRUCTURES = ("soft", "round", "oval", "sharp", "angular", "mature", "youthful", "severe")
COMBAT_ROLES = ("melee", "ranged", "support", "control", "tank", "assassin", "hybrid", "unknown")
WEAPON_TYPES = ("none", "scythe", "sword", "dagger", "spear", "bow", "wand", "shield", "axe")
ENERGY_TYPES = ("none", "arcane", "celestial", "elemental", "sound", "death", "tempo", "gravity", "demonic", "hybrid", "unknown", "custom")
ARCHETYPES = ("leader", "protector", "strategist", "rival", "berserker", "prodigy", "mentor", "healer", "guardian", "schemer", "tragic", "wanderer", "royal", "unknown")
VOICE_TONES = ("soft", "calm", "cold", "warm", "playful", "sharp", "aggressive", "broken", "elegant", "authoritative", "childlike")
MOVEMENT_STYLES = ("precise", "fluid", "elegant", "heavy", "fast", "unstable", "predatory", "disciplined", "brutal", "floating")
ETHNIC_INFLUENCES = ("none", "northern", "eastern", "southern", "western", "desert", "mountain", "islander", "celestial", "mixed", "fantasy_custom")
AURA_DENSITIES = ("light", "medium", "heavy", "overwhelming")
AURA_EFFECTS = ("flames", "particles", "light", "distortion", "smoke", "electricity", "custom")
CHARACTER_MOODS = ["happy", "neutral", "sad", "angry", "surprised", "scared", "tired", "home", "playful", "motivated", "shy", "jealous", "yandere", "crying"]
MOOD_VARIANTS = ["left", "default", "right"]

def _normalize_option_key(value: str) -> str:
    return re.sub(r"[\s-]+", "_", value.strip().lower())

def _normalize_enum_value(value, allowed_values, fallback: str, aliases: dict[str, str] | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback

    normalized_value = _normalize_option_key(value)
    for allowed_value in allowed_values:
        if _normalize_option_key(allowed_value) == normalized_value:
            return allowed_value

    if aliases and normalized_value in aliases:
        return aliases[normalized_value]

    return fallback

def _normalize_text_list(value) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;\n]+", value) if item.strip()]

    return []

def _normalize_optional_number(value):
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)

    if isinstance(value, str) and value.strip():
        try:
            parsed = float(value)
            return int(parsed) if parsed.is_integer() else parsed
        except ValueError:
            return None

    return None

def _pick_string(*values) -> str:
    for value in values:
        if isinstance(value, str):
            return value

    return ""

def _pick_optional_string(*values) -> str | None:
    value = _pick_string(*values).strip()
    return value or None

def _format_character_scene_label(value: str) -> str:
    words = re.split(r"[_\-\s]+", (value or "").strip())
    return " ".join(word[:1].upper() + word[1:] for word in words if word)

def _normalize_character_design(design: dict) -> dict:
    design = design if isinstance(design, dict) else {}
    hair = design.get("hair") if isinstance(design.get("hair"), dict) else {}
    eyes = design.get("eyes") if isinstance(design.get("eyes"), dict) else {}
    aura = design.get("aura") if isinstance(design.get("aura"), dict) else {}

    energy_source = design.get("energyType")
    raw_energy_values = energy_source if isinstance(energy_source, list) else [energy_source]
    normalized_energy_values = []
    for raw_energy_value in raw_energy_values:
        normalized_value = _normalize_enum_value(raw_energy_value, ENERGY_TYPES, "unknown")
        if normalized_value not in normalized_energy_values:
            normalized_energy_values.append(normalized_value)

    if not normalized_energy_values:
        normalized_energy_values = ["unknown"]

    return {
        "type": _normalize_enum_value(design.get("type"), CHARACTER_DESIGN_TYPES, "Desconocido", LEGACY_CHARACTER_TYPE_MAP),
        "ageReal": _normalize_optional_number(design.get("ageReal")),
        "ageAppearance": _normalize_optional_number(design.get("ageAppearance")),
        "skinTone": _normalize_enum_value(design.get("skinTone"), SKIN_TONES, "light"),
        "bodyType": _normalize_enum_value(design.get("bodyType"), BODY_TYPES, "slim"),
        "height": _normalize_enum_value(design.get("height"), HEIGHT_CATEGORIES, "average"),
        "hair": {
            "color": _pick_string(hair.get("color"), design.get("hairColor")),
            "secondaryColor": _pick_optional_string(hair.get("secondaryColor")),
            "length": _normalize_enum_value(hair.get("length"), HAIR_LENGTHS, "medium"),
            "texture": _normalize_enum_value(hair.get("texture"), HAIR_TEXTURES, "straight"),
            "style": _normalize_enum_value(hair.get("style"), HAIR_STYLES, "custom"),
            "description": _pick_string(hair.get("description"), design.get("hairDescription")),
        },
        "eyes": {
            "color": _pick_string(eyes.get("color"), design.get("eyesColor")),
            "secondaryColor": _pick_optional_string(eyes.get("secondaryColor")),
            "special": _normalize_enum_value(eyes.get("special"), EYE_SPECIALS, "none"),
            "intensity": _normalize_enum_value(eyes.get("intensity"), EYE_INTENSITIES, "medium"),
            "description": _pick_string(eyes.get("description")),
        },
        "faceStructure": _normalize_enum_value(design.get("faceStructure"), FACE_STRUCTURES, "oval"),
        "distinctiveFeatures": _normalize_text_list(design.get("distinctiveFeatures")),
        "combatRole": _normalize_enum_value(design.get("combatRole"), COMBAT_ROLES, "unknown"),
        "weapon": _normalize_enum_value(design.get("weapon"), WEAPON_TYPES, "none"),
        "energyType": normalized_energy_values,
        "personalityDetails": _pick_string(design.get("personalityDetails"), design.get("personalityDescription"), design.get("personality")),
        "archetype": _normalize_enum_value(design.get("archetype"), ARCHETYPES, "unknown"),
        "coreTrait": _pick_string(design.get("coreTrait")),
        "innerConflict": _pick_string(design.get("innerConflict")),
        "likes": _normalize_text_list(design.get("likes")),
        "hates": _normalize_text_list(design.get("hates")),
        "fears": _normalize_text_list(design.get("fears")),
        "voice": _normalize_enum_value(design.get("voice"), VOICE_TONES, "calm"),
        "movementStyle": _normalize_enum_value(design.get("movementStyle"), MOVEMENT_STYLES, "disciplined"),
        "aura": {
            "color": _pick_string(aura.get("color")),
            "secondaryColor": _pick_optional_string(aura.get("secondaryColor")),
            "density": _normalize_enum_value(aura.get("density"), AURA_DENSITIES, "light"),
            "effect": _normalize_enum_value(aura.get("effect"), AURA_EFFECTS, "custom"),
            "description": _pick_string(aura.get("description")),
        },
        "symbolicTheme": _pick_string(design.get("symbolicTheme")),
        "ethnicInfluence": _normalize_enum_value(design.get("ethnicInfluence"), ETHNIC_INFLUENCES, "none"),
        "extraInfo": _pick_string(design.get("extraInfo"), design.get("notes")),
    }

def _normalize_character_template(character_id: str, character_data: dict) -> dict:
    template = character_data.get("template")
    if not isinstance(template, dict):
        template = {}

    design = template.get("designCharacter")
    if not isinstance(design, dict):
        design = {}

    scene_label_source = template.get("sceneLabel") or character_data.get("name") or character_id
    preview_mood = template.get("previewMood") if template.get("previewMood") in CHARACTER_MOODS else "neutral"
    preview_variant = template.get("previewVariant") if template.get("previewVariant") in MOOD_VARIANTS else "default"

    return {
        "sceneLabel": _format_character_scene_label(str(scene_label_source)) or _format_character_scene_label(character_id),
        "previewMood": preview_mood,
        "previewVariant": preview_variant,
        "designCharacter": _normalize_character_design(design),
    }

def _normalize_character_data(character_id: str, character_data: dict) -> dict:
    normalized = dict(character_data) if isinstance(character_data, dict) else {}
    normalized["template"] = _normalize_character_template(character_id, normalized)
    return normalized

def _resolve_character_preview_candidates(char_dir: Path, character_id: str, mood: str, variant: str) -> list[Path]:
    if variant == "default":
        return sorted([
            file for file in char_dir.glob(f"{character_id}-{mood}-*")
            if not file.name.endswith(".import")
            and f"-{mood}-left-" not in file.name
            and f"-{mood}-right-" not in file.name
        ])

    return sorted([
        file for file in char_dir.glob(f"{character_id}-{mood}-{variant}-*")
        if not file.name.endswith(".import")
    ])

def _resolve_character_preview_path(character_id: str, mood: str | None = None, variant: str | None = None) -> Path | None:
    data = load_characters_json()
    character_data = data.get(character_id, {})
    template = _normalize_character_template(character_id, character_data if isinstance(character_data, dict) else {})

    resolved_mood = mood if mood in CHARACTER_MOODS else template["previewMood"]
    resolved_variant = variant if variant in MOOD_VARIANTS else template["previewVariant"]

    char_dir = YLBTM_CHARACTERS_DIR / character_id
    if not char_dir.exists():
        return None

    attempts = [
        (resolved_mood, resolved_variant),
        (resolved_mood, "default"),
        ("neutral", "default"),
    ]

    seen: set[tuple[str, str]] = set()
    for attempt_mood, attempt_variant in attempts:
        key = (attempt_mood, attempt_variant)
        if key in seen:
            continue
        seen.add(key)

        candidates = _resolve_character_preview_candidates(char_dir, character_id, attempt_mood, attempt_variant)
        if candidates:
            return candidates[0]

    return None

def load_characters_json() -> CharacterRootData:
    """Carga los datos de personajes desde characters.json."""
    if not os.path.exists(CHARACTERS_PATH):
        return {}
    try:
        with open(CHARACTERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {}

            normalized: CharacterRootData = {}
            changed = False
            for character_id, character_data in data.items():
                normalized_character = _normalize_character_data(character_id, character_data)
                normalized[character_id] = normalized_character
                if normalized_character != character_data:
                    changed = True

            if changed:
                save_characters_json(normalized)

            return normalized
    except:
        return {}
def save_characters_json(data: CharacterRootData):
    """Guarda los datos de personajes en characters.json."""
    with open(CHARACTERS_PATH, "w", encoding="utf-8") as f:
        # Usamos indent=4 para mantenerlo legible
        json.dump(data, f, indent=4, ensure_ascii=False)
    backup_file(CHARACTERS_PATH)
# --- ENDPOINTS CRUD DE PERSONAJES ---
@app.get("/editor/characters")
def get_characters():
    """Retorna el diccionario completo y plano de personajes."""
    return load_characters_json()
_CHARACTERS_TRANS = {"displayDescription": "description"}

@app.post("/editor/characters/{character_id}")
def create_character(character_id: str, character_data: dict):
    """Crea un nuevo personaje con una clave de ID única."""
    data = load_characters_json()
    if character_id in data:
        raise HTTPException(status_code=400, detail=f"El personaje ID '{character_id}' ya existe.")
    normalized_character = _normalize_character_data(character_id, character_data)
    data[character_id] = normalized_character
    save_characters_json(data)
    _ensure_translation_entry("characters", character_id, normalized_character, _CHARACTERS_TRANS)
    return {"status": "created", "character_id": character_id}

@app.put("/editor/characters/{character_id}")
def update_character(character_id: str, character_data: dict):
    """Actualiza un personaje específico por su ID."""
    data = load_characters_json()
    if character_id not in data:
        raise HTTPException(status_code=404, detail="Personaje no encontrado")
    old = data[character_id]
    normalized_character = _normalize_character_data(character_id, character_data)
    data[character_id] = normalized_character
    save_characters_json(data)
    _sync_es_translation("characters", character_id, old, normalized_character, _CHARACTERS_TRANS)
    return {"status": "updated", "character_id": character_id}

@app.delete("/editor/characters/{character_id}")
def delete_character(character_id: str):
    """Elimina un personaje específico por su ID."""
    data = load_characters_json()
    if character_id not in data:
        raise HTTPException(status_code=404, detail="Personaje no encontrado")
    del data[character_id]
    save_characters_json(data)
    _delete_translation_entry("characters", character_id)
    return {"status": "deleted", "character_id": character_id}

# --- MOOD IMAGES ---
YLBTM_CHARACTERS_DIR = Path(r"C:\repos\ylbtm\assets\characters")

@app.get("/editor/characters/{character_id}/preview-image")
def get_character_preview_image(character_id: str, mood: str | None = None, variant: str | None = None):
    preview_path = _resolve_character_preview_path(character_id, mood, variant)
    if preview_path is None:
        raise HTTPException(404, detail="Preview no encontrada")
    return FileResponse(str(preview_path))

@app.get("/editor/characters/{character_id}/moods")
def get_character_moods(character_id: str):
    """Returns existing mood images grouped by mood and variant."""
    char_dir = YLBTM_CHARACTERS_DIR / character_id
    if not char_dir.exists():
        return {}
    result = {}
    for mood in CHARACTER_MOODS:
        all_files = sorted([
            f.name for f in char_dir.glob(f"{character_id}-{mood}-*")
            if not f.name.endswith(".import")
        ])
        variants: dict = {"left": [], "default": [], "right": []}
        for fname in all_files:
            if f"-{mood}-left-" in fname:
                variants["left"].append(fname)
            elif f"-{mood}-right-" in fname:
                variants["right"].append(fname)
            else:
                variants["default"].append(fname)
        result[mood] = variants
    return result

@app.post("/editor/characters/{character_id}/mood/{mood}/{variant}")
async def upload_character_mood(character_id: str, mood: str, variant: str, file: UploadFile = File(...)):
    """Upload a mood image for a specific variant (left/default/right)."""
    if mood not in CHARACTER_MOODS:
        raise HTTPException(400, detail=f"Mood inválido. Debe ser uno de: {CHARACTER_MOODS}")
    if variant not in MOOD_VARIANTS:
        raise HTTPException(400, detail=f"Variant inválido. Debe ser uno de: {MOOD_VARIANTS}")
    char_dir = YLBTM_CHARACTERS_DIR / character_id
    char_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix.lower() if file.filename and Path(file.filename).suffix else ".png"
    if variant == "default":
        existing = [f for f in char_dir.glob(f"{character_id}-{mood}-*")
                    if not f.name.endswith(".import")
                    and f"-{mood}-left-" not in f.name
                    and f"-{mood}-right-" not in f.name]
        filename = f"{character_id}-{mood}-{len(existing) + 1}{ext}"
    else:
        existing = [f for f in char_dir.glob(f"{character_id}-{mood}-{variant}-*") if not f.name.endswith(".import")]
        filename = f"{character_id}-{mood}-{variant}-{len(existing) + 1}{ext}"
    dest = char_dir / filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    logging.info(f"Mood image saved: {dest}")
    return {"status": "ok", "filename": filename}

@app.get("/editor/characters/{character_id}/mood-image/{filename}")
def get_mood_image(character_id: str, filename: str):
    """Serve a character mood image file."""
    safe_filename = Path(filename).name
    path = YLBTM_CHARACTERS_DIR / character_id / safe_filename
    if not path.exists():
        raise HTTPException(404, detail="Imagen no encontrada")
    return FileResponse(str(path))

@app.delete("/editor/characters/{character_id}/mood-image/{filename}")
def delete_mood_image(character_id: str, filename: str):
    """Delete a character mood image file."""
    safe_filename = Path(filename).name
    path = YLBTM_CHARACTERS_DIR / character_id / safe_filename
    if not path.exists():
        raise HTTPException(404, detail="Imagen no encontrada")
    path.unlink()
    logging.info(f"Mood image deleted: {path}")
    return {"status": "deleted", "filename": safe_filename}

@app.put("/editor/characters/{character_id}/mood-image/{filename}")
async def replace_mood_image(character_id: str, filename: str, file: UploadFile = File(...)):
    """Replace an existing mood image file in place (same filename)."""
    safe_filename = Path(filename).name
    path = YLBTM_CHARACTERS_DIR / character_id / safe_filename
    if not path.exists():
        raise HTTPException(404, detail="Imagen no encontrada")
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)
    logging.info(f"Mood image replaced: {path}")
    return {"status": "replaced", "filename": safe_filename}

SPELLS_FILE = TARGET_DIR / "spells.json"

def load_spells():
    if not SPELLS_FILE.exists():
        return {}
    with open(SPELLS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
def save_spells(data):
    with open(SPELLS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    backup_file(SPELLS_FILE)
# --- ENDPOINTS SPELLS ---
@app.get("/editor/spells")
def get_spells():
    return load_spells()
@app.post("/editor/spells/{spell_id}")
def create_spell(spell_id: str, spell_data: dict):  
    data = load_spells()
    if spell_id in data:
        raise HTTPException(status_code=400, detail="ID de hechizo ya existe")
    data[spell_id] = spell_data
    save_spells(data)
    return {"status": "created", "id": spell_id}
@app.put("/editor/spells/{spell_id}")
def update_spell(spell_id: str, spell_data: dict):
    data = load_spells()
    data[spell_id] = spell_data
    save_spells(data)
    return {"status": "updated", "id": spell_id}
@app.delete("/editor/spells/{spell_id}")
def delete_spell(spell_id: str):
    data = load_spells()
    if spell_id not in data:
        raise HTTPException(status_code=404, detail="Hechizo no encontrado")
        
    del data[spell_id]
    save_spells(data)
    return {"status": "deleted", "id": spell_id}

SOULS_FILE = TARGET_DIR / "souls.json"
def load_souls():
    if not SOULS_FILE.exists():
        return {}
    with open(SOULS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
def save_souls(data):
    with open(SOULS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    backup_file(SOULS_FILE)
# --- ENDPOINTS SOULS ---
@app.get("/editor/souls")
def get_souls():
    return load_souls()
@app.post("/editor/souls/{soul_id}")
def create_soul(soul_id: str, soul_data: dict):  
    data = load_souls()
    if soul_id in data:
        raise HTTPException(status_code=400, detail="ID de soula ya existe")
    data[soul_id] = soul_data
    save_souls(data)
    return {"status": "created", "id": soul_id}
@app.put("/editor/souls/{soul_id}")
def update_soul(soul_id: str, soul_data: dict):
    data = load_souls()
    data[soul_id] = soul_data
    save_souls(data)
    return {"status": "updated", "id": soul_id} 
@app.delete("/editor/souls/{soul_id}")
def delete_soul(soul_id: str):
    data = load_souls()
    if soul_id not in data:
        raise HTTPException(status_code=404, detail="Soula no encontrada")
        
    del data[soul_id]
    save_souls(data)
    return {"status": "deleted", "id": soul_id}

MERCHANTS_FILE = TARGET_DIR / "merchants.json"

def load_merchants():
    if not MERCHANTS_FILE.exists():
        return {}
    with open(MERCHANTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_merchants(data):
    with open(MERCHANTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    backup_file(MERCHANTS_FILE)

_MERCHANTS_TRANS = {"displayDescription": "description"}

# --- ENDPOINTS MERCHANTS ---
@app.get("/editor/merchants")
def get_merchants():
    """Retorna el diccionario completo de merchants."""
    return load_merchants()

@app.post("/editor/merchants/{merchant_id}")
def create_merchant(merchant_id: str, merchant_data: dict):
    """Crea un nuevo merchant con una clave de ID única."""
    data = load_merchants()
    if merchant_id in data:
        raise HTTPException(status_code=400, detail=f"El merchant ID '{merchant_id}' ya existe.")
    data[merchant_id] = merchant_data
    save_merchants(data)
    _ensure_translation_entry("merchants", merchant_id, merchant_data, _MERCHANTS_TRANS)
    return {"status": "created", "merchant_id": merchant_id}

@app.put("/editor/merchants/{merchant_id}")
def update_merchant(merchant_id: str, merchant_data: dict):
    """Actualiza un merchant específico por su ID."""
    data = load_merchants()
    if merchant_id not in data:
        raise HTTPException(status_code=404, detail="Merchant no encontrado")
    old = data[merchant_id]
    data[merchant_id] = merchant_data
    save_merchants(data)
    _sync_es_translation("merchants", merchant_id, old, merchant_data, _MERCHANTS_TRANS)
    return {"status": "updated", "merchant_id": merchant_id}

@app.delete("/editor/merchants/{merchant_id}")
def delete_merchant(merchant_id: str):
    """Elimina un merchant específico por su ID."""
    data = load_merchants()
    if merchant_id not in data:
        raise HTTPException(status_code=404, detail="Merchant no encontrado")
    del data[merchant_id]
    save_merchants(data)
    _delete_translation_entry("merchants", merchant_id)
    return {"status": "deleted", "merchant_id": merchant_id}


# --- BACKGROUND IMAGES ---
YLBTM_BACKGROUNDS_DIR = Path(r"C:\repos\ylbtm\assets\backgrounds")

@app.get("/editor/backgrounds")
def list_backgrounds():
    """List all background images."""
    YLBTM_BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        f.name for f in sorted(YLBTM_BACKGROUNDS_DIR.iterdir())
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        and not f.name.endswith(".import")
    ]
    return {"backgrounds": files}

@app.post("/editor/backgrounds")
async def upload_background(file: UploadFile = File(...)):
    """Upload a new background image, auto-naming it background_N.ext."""
    YLBTM_BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix.lower() if file.filename and Path(file.filename).suffix else ".png"
    existing_stems = {
        f.stem for f in YLBTM_BACKGROUNDS_DIR.iterdir()
        if f.is_file() and not f.name.endswith(".import")
    }
    n = 1
    while f"background_{n}" in existing_stems:
        n += 1
    filename = f"background_{n}{ext}"
    dest = YLBTM_BACKGROUNDS_DIR / filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    logging.info(f"Background saved: {dest}")
    return {"status": "ok", "filename": filename}

@app.get("/editor/backgrounds/{filename}")
def get_background(filename: str):
    """Serve a background image."""
    safe_name = Path(filename).name
    path = YLBTM_BACKGROUNDS_DIR / safe_name
    if not path.resolve().parent == YLBTM_BACKGROUNDS_DIR.resolve() or not path.exists():
        raise HTTPException(404, detail="Background not found")
    return FileResponse(str(path))

@app.delete("/editor/backgrounds/{filename}")
def delete_background(filename: str):
    """Delete a background image."""
    safe_name = Path(filename).name
    path = YLBTM_BACKGROUNDS_DIR / safe_name
    if not path.exists():
        raise HTTPException(404, detail="Background not found")
    path.unlink()
    logging.info(f"Background deleted: {path}")
    return {"status": "deleted", "filename": safe_name}