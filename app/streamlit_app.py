"""¿Qué mascota es? · Clasificador de razas de perros y gatos (Streamlit).

Ejecutar localmente desde la raíz del proyecto:
    uv run streamlit run app/streamlit_app.py

Modelo: ResNet18 ajustado sobre Oxford-IIIT Pet (ver la sección de créditos).
La inferencia usa ONNX Runtime, así que no necesita PyTorch en el servidor.
"""
import json
import math
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "best.onnx"
META_PATH = APP_DIR / "model" / "classes.json"
METRICS_PATH = APP_DIR / "assets" / "metrics.json"
CONFUSION_PATH = APP_DIR / "assets" / "confusion_matrix.png"

# Mismos valores que cv/data.py (ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Escala de confianza. Se aplica al porcentaje ya redondeado a 1 decimal,
# para que el color siempre coincida con el número que se muestra.
SCALE = [
    {"max": 40, "key": "low", "name": "Baja", "color": "#e5484d",
     "note": "La red duda; prueba con otra foto."},
    {"max": 75, "key": "mid", "name": "Media", "color": "#2f6fe4",
     "note": "Respuesta razonable, pero hay otras razas en juego."},
    {"max": 101, "key": "high", "name": "Alta", "color": "#1fa463",
     "note": "La red está muy segura de su respuesta."},
]

# Sprites pixel art: cada letra es un color de la paleta ("." es transparente)
SPRITES = {
    "cat": {
        "palette": {"K": "#4a3f5c", "o": "#ffc48a", "e": "#3d3d5c", "w": "#ffffff",
                    "p": "#ffb3c7", "n": "#ff8fab"},
        "rows": [
            "..KK............KK..",
            "..KpK..........KpK..",
            ".KppKKKKKKKKKKKKppK.",
            ".KooooooooooooooooK.",
            ".KoweooooooooooewoK.",
            ".KoeeooooooooooeeoK.",
            ".KppooooooooooooppK.",
            ".KooooooonnoooooooK.",
            "..KooooooooooooooK..",
            ".KooooooooooooooooK.",
            "..KooooooooooooooK..",
            "...KKKKKKKKKKKKKK...",
        ],
    },
    "dog": {
        "palette": {"K": "#4a3f5c", "b": "#e8c39e", "E": "#c99a6b", "k": "#3d3d5c",
                    "w": "#ffffff", "p": "#ffb3c7"},
        "rows": [
            "...KKKKKKKKKKKKKK...",
            "..KbbbbbbbbbbbbbbK..",
            "KEEbbbbbbbbbbbbbbEEK",
            "KEEbkwbbbbbbbbwkbEEK",
            "KEEbkkbbbbbbbbkkbEEK",
            "KEEbbbbbbbbbbbbbbEEK",
            "KEEbbbbbkkkkbbbbbEEK",
            "KEbbbbbbbbbbbbbbbbEK",
            ".KbbbbbbppppbbbbbbK.",
            "..KKKKKKKKKKKKKKKK..",
        ],
    },
    "plant": {
        "palette": {"G": "#8fd6a8", "g": "#bfe8cf", "K": "#4a3f5c", "P": "#f7a8c0"},
        "rows": [
            ".......GG.......",
            "....GGGgGGG.....",
            "..GGGgGGgGGGG...",
            ".GGgGGGGGGGgGG..",
            "...GGGGGGGGG....",
            "....GGGGGGG.....",
            "....KKKKKKK.....",
            "...KPPPPPPPK....",
            "...KPPPPPPPK....",
            "....KKKKKKK.....",
        ],
    },
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Nunito:wght@400;700;800&display=swap');

.stApp {
  background: linear-gradient(180deg, #fdf1f5, #eef6ff);
  font-family: "Nunito", system-ui, sans-serif;
  color: #2d2a4a;
}
h1, h2, h3 {
  font-family: "Press Start 2P", monospace !important;
  font-weight: 400 !important;
  line-height: 1.6 !important;
}
h1 { font-size: 22px !important; }
h2 { font-size: 14px !important; }
h3 { font-size: 11px !important; }

/* Secciones: tarjetas estilo pixel */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: #fffdf7;
  border: 4px solid #2d2a4a !important;
  border-radius: 0 !important;
  box-shadow: 6px 6px 0 #2d2a4a;
  padding: 8px 12px;
  margin-bottom: 26px;
}

.muted { opacity: 0.75; font-size: 14px; }

.step {
  border: 3px solid #2d2a4a;
  background: #fff;
  padding: 12px;
  min-height: 220px;
}
.step .num {
  display: inline-block;
  width: 32px; height: 32px; line-height: 26px;
  text-align: center;
  border: 3px solid #2d2a4a;
  background: #fff3c4;
  font-family: "Press Start 2P", monospace;
  font-size: 11px;
  margin-bottom: 8px;
}
.step p { font-size: 14px; margin: 0; }

.pct {
  display: inline-block;
  font-family: "Press Start 2P", monospace;
  font-size: 13px;
  padding: 6px 8px;
  border: 3px solid #2d2a4a;
  color: #fff;
  box-shadow: 3px 3px 0 #2d2a4a;
}
.pct.low { background: #e5484d; }
.pct.mid { background: #2f6fe4; }
.pct.high { background: #1fa463; }

.bar {
  height: 14px;
  border: 3px solid #2d2a4a;
  background: #fff;
  margin: 8px 0 14px 0;
}
.bar > div { height: 100%; }

.bubble {
  display: inline-block;
  background: #fff;
  border: 3px solid #2d2a4a;
  padding: 10px 12px;
  font-weight: 800;
}
.advice {
  font-size: 14px;
  padding: 10px 12px;
  border: 3px solid #2d2a4a;
  background: #ffe0cc;
}
.top3-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-weight: 800; }
.legend-item { border: 3px solid #2d2a4a; background: #fff; padding: 12px; height: 100%; }
.legend-item p { font-size: 14px; margin: 8px 0 0 0; }

.footer-text {
  font-family: "Press Start 2P", monospace;
  font-size: 11px;
  text-align: center;
  padding: 18px 12px;
}
.grass {
  height: 26px;
  border-top: 4px solid #2d2a4a;
  background: repeating-linear-gradient(90deg, #d4f1e3 0 16px, #bfe8cf 16px 32px);
  margin-top: 30px;
}
</style>
"""


# ---------- Sprites ----------
@lru_cache(maxsize=None)
def sprite_png(name: str, scale: int) -> bytes:
    """Dibuja un sprite pixel art a PNG a partir de sus filas de letras."""
    rows, palette = SPRITES[name]["rows"], SPRITES[name]["palette"]
    img = Image.new("RGBA", (len(rows[0]) * scale, len(rows) * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == ".":
                continue
            draw.rectangle([x * scale, y * scale,
                            (x + 1) * scale - 1, (y + 1) * scale - 1],
                           fill=palette[ch])
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- Modelo ----------
@st.cache_resource(show_spinner="Cargando el modelo…")
def load_model():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return session, meta


def preprocess(img: Image.Image, size: int, short_side: int):
    """Espejo de cv/data.py: Resize(short_side) -> CenterCrop(size) -> Normalize."""
    w, h = img.size
    if w <= h:
        nw, nh = short_side, int(short_side * h / w)
    else:
        nh, nw = short_side, int(short_side * w / h)
    img = img.resize((nw, nh), Image.BILINEAR)
    left = round((nw - size) / 2)
    top = round((nh - size) / 2)
    crop = img.crop((left, top, left + size, top + size))

    arr = np.asarray(crop, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = arr.transpose(2, 0, 1)[None].astype(np.float32)
    return crop, tensor


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@st.cache_data(show_spinner="Analizando la foto…")
def analyze(image_bytes: bytes) -> dict:
    session, meta = load_model()
    img = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
    crop, tensor = preprocess(img, meta["image_size"], meta["resize_short_side"])

    logits = session.run(None, {"input": tensor})[0][0]
    probs = softmax(logits)

    species_p = {"perro": 0.0, "gato": 0.0}
    for p, s in zip(probs, meta["species"]):
        species_p[s] += float(p)
    species = max(species_p, key=species_p.get)

    order = np.argsort(-probs)
    top3 = [{"breed": meta["classes"][i], "species": meta["species"][i], "p": float(probs[i])}
            for i in order[:3]]

    buf = BytesIO()
    crop.save(buf, format="PNG")
    return {"species": species, "species_p": species_p, "top3": top3,
            "crop_png": buf.getvalue()}


# ---------- Escala y formato ----------
def pct_of(p: float) -> float:
    # Igual que Math.round(p * 1000) / 10 en JS: redondeo a 1 decimal
    return math.floor(p * 1000 + 0.5) / 10


def scale_for(p: float) -> dict:
    pct = pct_of(p)
    return next(s for s in SCALE if pct < s["max"])


def badge(p: float) -> str:
    s = scale_for(p)
    return f'<span class="pct {s["key"]}">{pct_of(p):.1f} %</span>'


def bar(p: float) -> str:
    s = scale_for(p)
    return (f'<div class="bar"><div style="width:{pct_of(p)}%;'
            f'background:{s["color"]}"></div></div>')


# ---------- Página ----------
st.set_page_config(page_title="¿Qué mascota es?", page_icon="🐾", layout="wide")
st.html(CSS)

if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0


def reset():
    st.session_state.reset_count += 1


# Cabecera
head_text, head_sprites = st.columns([3, 2])
with head_text:
    st.title("¿Qué mascota es?")
    st.markdown(
        '<p class="muted">Sube una foto de un perro o un gato y descubre su especie, '
        'su raza y qué tan segura está la red.</p>',
        unsafe_allow_html=True)
with head_sprites:
    c1, c2, c3 = st.columns(3)
    c1.image(sprite_png("plant", 5))
    c2.image(sprite_png("cat", 5))
    c3.image(sprite_png("dog", 5))

# 1. Cómo funciona
with st.container(border=True):
    st.subheader("1 · ¿Cómo funciona?")
    st.markdown(
        '<p class="muted">Cinco pasos, los mismos del notebook '
        '<code>notebooks/01_tutorial_clasificacion_mascotas.ipynb</code>.</p>',
        unsafe_allow_html=True)
    steps = [
        ("Subes tu foto", "La app la procesa en el servidor y no la guarda en disco."),
        ("Recorte 224×224", "La red solo mira un cuadrado centrado. Abajo ves exactamente lo que recibe."),
        ("Normalizamos", "Restamos la media de ImageNet y dividimos por la desviación, como en entrenamiento."),
        ("ResNet18", "La red busca rasgos (orejas, hocico, pelaje) y da 37 puntajes, uno por raza."),
        ("Softmax → %", "Los puntajes pasan a probabilidades que suman 100 %. La especie es la suma de sus razas."),
    ]
    for n, (col, (title, text)) in enumerate(zip(st.columns(5), steps), start=1):
        with col:
            st.markdown(
                f'<div class="step"><span class="num">{n}</span>'
                f'<h3>{title}</h3><p>{text}</p></div>',
                unsafe_allow_html=True)

# 2. Subida
with st.container(border=True):
    st.subheader("2 · Sube tu foto")
    uploaded = st.file_uploader(
        "Arrastra una foto aquí o haz clic para elegirla (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"uploader_{st.session_state.reset_count}",
    )

result = None
if uploaded is not None:
    data = uploaded.getvalue()
    result = analyze(data)
    original = ImageOps.exif_transpose(Image.open(BytesIO(data)))

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(original, caption="Tu foto", width=420)
    with col_b:
        st.image(result["crop_png"], caption="Lo que ve la red (224×224)", width=224)

# 3. Resultado
if result is not None:
    species = result["species"]
    sp_p = result["species_p"][species]
    top = result["top3"]
    breed = top[0]
    breed_scale = scale_for(breed["p"])
    species_label = "Gato 🐱" if species == "gato" else "Perro 🐶"
    bubble = "¡Creo que es un gato!" if species == "gato" else "¡Creo que es un perro!"

    with st.container(border=True):
        st.subheader("3 · Resultado")
        left, right = st.columns([1, 1], gap="large")

        with left:
            m_col, b_col = st.columns([1, 2])
            m_col.image(sprite_png("cat" if species == "gato" else "dog", 8), width=160)
            b_col.markdown(f'<div class="bubble">{bubble}</div>', unsafe_allow_html=True)

            st.markdown("**Especie**")
            st.markdown(f"### {species_label}")
            st.markdown(badge(sp_p), unsafe_allow_html=True)
            st.markdown(bar(sp_p), unsafe_allow_html=True)

            st.markdown("**Raza más probable**")
            st.markdown(f"### {breed['breed']}")
            st.markdown(badge(breed["p"]), unsafe_allow_html=True)
            st.markdown(bar(breed["p"]), unsafe_allow_html=True)
            st.markdown(
                f'<div class="advice">Confianza {breed_scale["name"].lower()}: '
                f'{breed_scale["note"]}</div>',
                unsafe_allow_html=True)

        with right:
            st.markdown("**Top 3 razas**")
            for i, t in enumerate(top, start=1):
                st.markdown(
                    f'<div class="top3-row"><span>{i}. {t["breed"]} '
                    f'<small class="muted">({t["species"]})</small></span>{badge(t["p"])}</div>'
                    f'{bar(t["p"])}',
                    unsafe_allow_html=True)
            st.markdown(
                '<p class="muted">La suma de las 37 razas siempre es 100 %. Si la foto no '
                'es de perro o gato, el resultado será poco confiable.</p>',
                unsafe_allow_html=True)

    # 5. Acciones (se muestran junto al resultado)
    with st.container(border=True):
        st.subheader("4 · Acciones del proyecto")
        st.markdown(
            '<p class="muted">Herramientas para llevar este resultado al proyecto de Python.</p>',
            unsafe_allow_html=True)

        payload = {
            "file": uploaded.name,
            "species": {"name": species, "probability": sp_p},
            "breed": {"name": breed["breed"], "probability": breed["p"]},
            "top3": top,
            "scale": breed_scale["name"],
        }
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("**Predecir desde la terminal**")
            st.code(f'uv run python -m cv.predict "{uploaded.name}" --topk 3', language="bash")
            st.download_button(
                "Descargar resultado (.json)",
                data=json.dumps(payload, ensure_ascii=False, indent=2),
                file_name=f"resultado_{Path(uploaded.name).stem}.json",
                mime="application/json",
            )
        with a2:
            st.markdown("**Entrenar de nuevo**")
            st.code("uv run python -m cv.train --epochs 3", language="bash")
            st.button("Otra foto", on_click=reset)

        with st.expander("Métricas del modelo (conjunto de test)"):
            if METRICS_PATH.exists():
                m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                mc1, mc2 = st.columns(2)
                mc1.metric("Test accuracy", f"{m['test_accuracy'] * 100:.1f} %")
                mc2.metric("F1 macro", f"{m['macro_f1'] * 100:.1f} %")
            else:
                st.info("No encontré metrics.json. Corre: uv run python -m cv.evaluate")

        with st.expander("Matriz de confusión"):
            if CONFUSION_PATH.exists():
                st.image(str(CONFUSION_PATH), use_container_width=True)
            else:
                st.info("No encontré la matriz. Corre: uv run python -m cv.evaluate")

# Si todavía no hay foto, muestra las métricas igual en la sección de acciones
if result is None:
    with st.container(border=True):
        st.subheader("3 · Resultado")
        st.markdown('<p class="muted">Sube una foto en la sección 2 para ver el resultado.</p>',
                    unsafe_allow_html=True)

# 4. Escala de confianza
with st.container(border=True):
    st.subheader("5 · Escala de confianza")
    st.markdown('<p class="muted">Cada porcentaje se colorea según qué tan segura está la red.</p>',
                unsafe_allow_html=True)
    lg = st.columns(3)
    texts = [
        ("0–39 %", "low", "<strong>Baja · rojo.</strong> La red duda. Probablemente la foto es confusa "
                          "o la raza se parece a otra."),
        ("40–74 %", "mid", "<strong>Media · azul.</strong> Respuesta razonable, pero hay otras razas en juego."),
        ("75–100 %", "high", "<strong>Alta · verde.</strong> La red está muy segura de su respuesta."),
    ]
    for col, (label, key, text) in zip(lg, texts):
        with col:
            st.markdown(f'<div class="legend-item"><span class="pct {key}">{label}</span>'
                        f'<p>{text}</p></div>', unsafe_allow_html=True)

# Créditos
with st.container(border=True):
    st.subheader("Créditos y licencias")
    st.markdown(
        """
**Dataset.** O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. *Cats and Dogs*.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
Disponible en [robots.ox.ac.uk/~vgg/data/pets](https://www.robots.ox.ac.uk/~vgg/data/pets/).
Licencia **CC BY-SA 4.0**. Los derechos de las imágenes pertenecen a sus autores originales.

**Arquitectura.** K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition*.
CVPR, 2016. Pesos iniciales de ResNet18 preentrenados en ImageNet (vía torchvision).

**Modelo.** Ajustado sobre Oxford-IIIT Pet. Por ser una obra derivada del dataset, conserva la
licencia CC BY-SA 4.0 y esta cita.
        """
    )

st.markdown(
    '<div class="footer-text">Con &lt;3 &lt;3 &lt;3 para la Dragoncita rica.</div>'
    '<div class="grass"></div>',
    unsafe_allow_html=True)
