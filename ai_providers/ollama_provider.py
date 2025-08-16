
import os, requests

BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

def ollama_generate(prompt: str, system: str|None=None, **kwargs):
    payload = {
        "model": MODEL,
        "prompt": f"{('System: '+system+'\n') if system else ''}{prompt}",
        "stream": False,
        "options": {"temperature": kwargs.get("temperature", 0.2)}
    }
    r = requests.post(f"{BASE}/api/generate", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("response","").strip(), {"model": MODEL}
