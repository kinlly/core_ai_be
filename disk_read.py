import json
from pathlib import Path
import os

def load_jsonl(path: Path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_main_jsonl():
    return load_jsonl(os.getenv("MAIN_DATA_PATH"))