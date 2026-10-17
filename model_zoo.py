"""Fine-tune edilecek VL modellerinin kaydi. Anahtar (key) tum scriptlerde `--model` olarak kullanilir.

response_marker: asistan cevabinin basladigi chat-template metni; egitimde bu isaretten ONCEKI tokenlar loss'tan maskelenir.
nested_images: processor'a goruntuler `[[img], [img], ...]` seklinde mi verilir (Qwen: duz liste).
"""

MODELS = {
    "qwen25vl_7b": dict(
        hf_id="Qwen/Qwen2.5-VL-7B-Instruct",
        family="qwen",
        response_marker="<|im_start|>assistant\n",
        nested_images=False,
        dynamic_resolution=True,   # min_pixels/max_pixels processor argumani
        min_transformers="4.49.0",
    ),
    "qwen3vl_8b": dict(
        hf_id="Qwen/Qwen3-VL-8B-Instruct",
        family="qwen",
        response_marker="<|im_start|>assistant\n",
        nested_images=False,
        dynamic_resolution=True,
        min_transformers="4.57.0",
    ),
    # sonraki asama (notebook'lari henuz yok)
    "gemma3_12b": dict(
        hf_id="google/gemma-3-12b-it",
        family="gemma",
        response_marker="<start_of_turn>model\n",
        nested_images=True,
        dynamic_resolution=False,  # goruntu 896x896'ya sabitlenir
        min_transformers="4.50.0",
        gated=True,
    ),
    "llama32_11b": dict(
        hf_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        family="llama",
        response_marker="<|start_header_id|>assistant<|end_header_id|>\n\n",
        nested_images=True,
        dynamic_resolution=False,  # 560px tile, en fazla 4 tile
        min_transformers="4.45.0",
        gated=True,
    ),
    "internvl35_8b": dict(
        hf_id="OpenGVLab/InternVL3_5-8B-HF",
        family="internvl",
        response_marker="<|im_start|>assistant\n",
        nested_images=False,
        dynamic_resolution=False,
        min_transformers="4.52.0",
    ),
}

# Qwen ailesi bu asamada hazir; digerleri icin script uyumlulugu ayrica test edilecek
READY = ["qwen25vl_7b", "qwen3vl_8b"]


def get(key):
    if key not in MODELS:
        raise SystemExit(f"Bilinmeyen model '{key}'. Secenekler: {', '.join(MODELS)}")
    return MODELS[key]
