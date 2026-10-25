"""Exporta el mejor checkpoint a ONNX para usarlo en el navegador (carpeta web/).

Uso:
    uv run python -m cv.export
"""
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from cv.config import ROOT, load_config, resolve
from cv.model import load_checkpoint
from cv.species import species_of

# La web estática y la app de Streamlit comparten este modelo
MODEL_DIR = ROOT / "app" / "model"


def export(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    model, ckpt = load_checkpoint(resolve(cfg["train"]["checkpoint"]), torch.device("cpu"))
    size = ckpt["image_size"]
    classes = ckpt["classes"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODEL_DIR / "best.onnx"
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model, (dummy,), onnx_path,
        input_names=["input"], output_names=["logits"],
        dynamo=True,
    )

    # El exportador nuevo guarda los pesos en best.onnx.data. Los volvemos a
    # meter en un solo archivo para que la web solo tenga que descargar uno.
    proto = onnx.load(onnx_path)
    onnx.save_model(proto, onnx_path, save_as_external_data=False)
    (MODEL_DIR / "best.onnx.data").unlink(missing_ok=True)

    # Metadatos que necesita la web: nombres, especie y el preprocesado exacto.
    meta = {
        "classes": classes,
        "species": [species_of(c) for c in classes],
        "image_size": size,
        "resize_short_side": int(size * 1.14),
    }
    (MODEL_DIR / "classes.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Comprobación: ONNX debe dar los mismos logits que PyTorch.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())

    print(f"ONNX guardado en {onnx_path.relative_to(ROOT)} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Diferencia máxima PyTorch vs ONNX: {max_diff:.2e}")


if __name__ == "__main__":
    export()
