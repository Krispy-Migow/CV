# CV – Clasificación de razas de mascotas

Proyecto de ejemplo de Computer Vision: clasifica **37 razas de perros y gatos** (dataset Oxford-IIIT Pet) usando **transfer learning** con una ResNet18 preentrenada en ImageNet.

Incluye:

- Código modular en `src/cv/` (datos, modelo, entrenamiento, evaluación, inferencia).
- Un notebook tutorial paso a paso: `notebooks/01_tutorial_clasificacion_mascotas.ipynb`.
- Dependencias gestionadas con **uv**.

## Requisitos

- [uv](https://docs.astral.sh/uv/) (`pip install uv` o `winget install astral-sh.uv`).
- Python 3.12 (uv lo descarga si no lo tienes).
- GPU NVIDIA opcional. El proyecto instala PyTorch con CUDA 12.8 (ver `pyproject.toml`).

## Instalación

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Para CPU, elimina la sección `[tool.uv.index]` y `[tool.uv.sources]` de `pyproject.toml` y ejecuta `uv sync` de nuevo.

## Uso

Todas las rutas e hiperparámetros están en `configs/config.yaml`.

```bash
# 1. Entrenar (descarga el dataset la primera vez, ~800 MB)
uv run python -m cv.train --epochs 3

# 2. Evaluar en el conjunto de test (métricas y matriz de confusión en outputs/)
uv run python -m cv.evaluate

# 3. Predecir la raza de una imagen (top-3)
uv run python -m cv.predict ruta/a/tu_foto.jpg --topk 3

# 4. Pruebas
uv run pytest
```

## App en Streamlit (para compartir un enlace)

`app/streamlit_app.py` es la versión lista para publicar. Usa ONNX Runtime, así que no necesita PyTorch en el servidor.

```bash
uv run streamlit run app/streamlit_app.py
```

Para desplegarla en Streamlit Community Cloud, `app/requirements.txt` define las dependencias y el archivo principal es `app/streamlit_app.py`. El modelo (`app/model/best.onnx`, unos 45 MB) y las métricas ya están dentro de `app/`.

Si re-entrenas, `uv run python -m cv.export` escribe el modelo directamente en `app/model/`. Las métricas (`app/assets/metrics.json`) se actualizan copiando `outputs/metrics.json` y `outputs/confusion_matrix.png` a `app/assets/`.

## Dataset y licencia

Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). *Cats and Dogs*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Disponible en <https://www.robots.ox.ac.uk/~vgg/data/pets/>.

Licencia del dataset: **CC BY-SA 4.0**. El modelo es una obra derivada del dataset, así que conserva esa licencia.

## Página web (subir tu foto)

Exporta el modelo a ONNX (una sola vez, o cada vez que re-entrenes) y levanta un servidor estático desde la raíz del proyecto:

```bash
uv run python -m cv.export
uv run python -m http.server 8000
```

Abre [http://localhost:8000/web/](http://localhost:8000/web/) y sube una foto. La inferencia corre en tu navegador con `onnxruntime-web`; la imagen no sale de tu equipo.

- La especie es la suma de las probabilidades de sus razas (12 gatos, 25 perros).
- Escala de confianza: **0–39 % rojo** (baja), **40–74 % azul** (media), **75–100 % verde** (alta).
- Pequeña diferencia con la CLI: el redimensionado del navegador (canvas) no es idéntico al de torchvision, así que los porcentajes pueden variar un par de puntos.
- Los botones de la sección 5 copian comandos, descargan el resultado en JSON y abren las métricas y la matriz de confusión del proyecto.

### Publicar la página en Netlify

El repo incluye `netlify.toml`: publica la raíz sin paso de build y redirige `/` a `web/index.html`. En Netlify: *Add new site → Import an existing project*, elige el repo y no cambies la configuración. La página lee el modelo desde `app/model/`, así que no hay que copiar archivos.

Para el paso a paso comentado, abre el notebook:

```bash
uv run jupyter lab notebooks/01_tutorial_clasificacion_mascotas.ipynb
```

## Estructura

```
configs/config.yaml        Configuración (rutas, batch size, épocas, lr...)
src/cv/
  config.py                Carga de configuración y rutas
  data.py                  Transformaciones, datasets y DataLoaders
  model.py                 ResNet18 con cabeza nueva y carga de checkpoints
  train.py                 Bucle de entrenamiento -> models/best.pt
  evaluate.py              Accuracy, F1 y matriz de confusión -> outputs/
  predict.py               Inferencia top-k sobre una imagen
notebooks/                 Tutorial paso a paso
tests/                     Pruebas rápidas del modelo
data/raw/                  Dataset descargado (no versionado)
models/                    Checkpoints (no versionados)
outputs/                   Métricas e imágenes generadas (no versionadas)
```

## Resultados de referencia

Con 3 épocas en una RTX 3060 Laptop (solo se entrena la capa final):

| Métrica          | Valor  |
|------------------|--------|
| Val accuracy     | 0.864  |
| Test accuracy    | 0.842  |
| Test macro F1    | 0.838  |

Los valores pueden variar ligeramente según el hardware y la semilla.
