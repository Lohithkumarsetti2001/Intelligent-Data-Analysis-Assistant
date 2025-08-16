
import os, requests

def hf_generate(prompt: str, system: str|None=None, **kwargs):
    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise RuntimeError("HF_API_TOKEN not set")
    # Using text-generation-inference route compatible endpoints
    model = kwargs.get("hf_model", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": f"{('System: '+system+'\n') if system else ''}{prompt}",
        "parameters": {"max_new_tokens": 250, "temperature": kwargs.get("temperature",0.2)}
    }
    resp = requests.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        txt = data[0]["generated_text"]
    else:
        # some endpoints return dict
        txt = data.get("generated_text", str(data))
    return txt.strip(), {"model": model}
