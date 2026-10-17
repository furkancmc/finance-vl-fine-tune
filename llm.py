"""Evren LLM API (OpenAI uyumlu) icin kucuk CLI sarmalayici.

Kullanim:
    python llm.py "istek metni"
    python llm.py "istek" -f kap_collect.py -f baska.py        # dosyalari baglam olarak ekler
    python llm.py "istek" -o yeni_dosya.py                      # cevabi dosyaya yazar (kod blogu varsa sadece kodu)
    python llm.py "istek" -s "sistem mesaji"

Ortam degiskenleri: EVREN_API_KEY (zorunlu), EVREN_BASE_URL, EVREN_MODEL
"""
import argparse
import os
import re
import sys
from pathlib import Path

import requests

BASE_URL = os.environ.get("EVREN_BASE_URL", "https://evren-llmapi.ssyz.org.tr/v1")
MODEL = os.environ.get("EVREN_MODEL", "deepseek/deepseek-v4.1-flash")


def chat(messages, model=MODEL, temperature=0.2, max_tokens=16000):
    key = os.environ.get("EVREN_API_KEY")
    if not key:
        sys.exit("EVREN_API_KEY ayarli degil.")
    r = requests.post(f"{BASE_URL.rstrip('/')}/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "messages": messages, "temperature": temperature,
                            "max_tokens": max_tokens},
                      timeout=600)
    if not r.ok:
        sys.exit(f"API hatasi {r.status_code}: {r.text[:500]}")
    return r.json()["choices"][0]["message"]["content"]


def extract_code(text):
    # ilk citten son cite kadar (kodun icindeki ic citler kesmesin)
    m = re.search(r"```[\w+-]*\n(.*)```", text, re.S)
    return m.group(1) if m else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-f", "--file", action="append", default=[], help="baglam olarak eklenecek dosya")
    ap.add_argument("-o", "--out", help="cevabin yazilacagi dosya")
    ap.add_argument("-s", "--system", default="Sen deneyimli bir yazilim muhendisisin. Kisa ve dogru cevap ver.")
    ap.add_argument("-m", "--model", default=MODEL)
    a = ap.parse_args()

    ctx = "".join(f"\n\n--- {f} ---\n{Path(f).read_text(encoding='utf-8')}" for f in a.file)
    reply = chat([{"role": "system", "content": a.system},
                  {"role": "user", "content": a.prompt + ctx}], model=a.model)
    if a.out:
        Path(a.out).write_text(extract_code(reply), encoding="utf-8")
        print(f"yazildi: {a.out}")
    else:
        print(reply)


if __name__ == "__main__":
    main()
