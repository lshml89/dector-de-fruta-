import streamlit as st
from PIL import Image
import os
import glob
import numpy as np
from utils.camera_utils import Classifier

# -------------------------------
# Configuración de la página
# -------------------------------
st.set_page_config(page_title="🍎 Clasificador de Frutas", page_icon="🍌", layout="centered")
st.title("🍎 Clasificador de Frutas")
st.caption("Sube una foto o usa la cámara; detectará si es Plátano 🍌 o Naranja 🍊 usando agrupamiento no supervisado.")

MODEL_FOLDER = "modelos"
MIN_CONF = 0.50  # umbral mínimo para aceptar detección

# -------------------------------
# Asignación manual de grupos a frutas
# -------------------------------
def grupo_a_fruta(idx: int) -> str:
    if idx == 0:
        return "Plátano 🍌"
    elif idx == 1:
        return "Naranja 🍊"
    else:
        return f"Grupo {idx}"

# -------------------------------
# Cargar modelos .tflite
# -------------------------------
model_paths = sorted(glob.glob(os.path.join(MODEL_FOLDER, "*.tflite")))
if not model_paths:
    st.error(f"⚠️ No se encontraron modelos .tflite en '{MODEL_FOLDER}'")
    st.stop()

classifiers = []
for mp in model_paths:
    try:
        clf = Classifier(mp)
        classifiers.append({
            "path": mp,
            "name": os.path.basename(mp).replace(".tflite", ""),
            "clf": clf
        })
    except Exception as e:
        st.warning(f"No se pudo cargar el modelo '{mp}': {e}")

if not classifiers:
    st.error("⚠️ No se pudo inicializar ningún clasificador.")
    st.stop()

# -------------------------------
# UI: cámara / upload
# -------------------------------
col1, col2 = st.columns([1, 1])
with col1:
    modo = st.radio("Entrada:", ["📷 Cámara", "📁 Subir imagen"])
    if modo == "📷 Cámara":
        cam_file = st.camera_input("Toma una foto")
        upload_file = None
    else:
        cam_file = None
        upload_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

image_file = cam_file if cam_file is not None else upload_file

# -------------------------------
# Predicción y visualización
# -------------------------------
if image_file:
    try:
        img = Image.open(image_file).convert("RGB")
    except Exception as e:
        st.error(f"No se pudo abrir la imagen: {e}")
        st.stop()

    detecciones = []

    for entry in classifiers:
        clf = entry["clf"]
        model_name = entry["name"]

        try:
            pred = clf.predict(img, etiquetas=None, guardar_csv=False, umbral=0.0)
        except Exception as e:
            st.warning(f"Error prediciendo con {model_name}: {e}")
            continue

        raw_out = pred.get("raw_output")
        if raw_out is None:
            raw_out = np.array([pred.get("confianza", 0.0)])
        raw_out = np.asarray(raw_out).flatten()
        best_idx = int(np.argmax(raw_out))
        score = float(raw_out[best_idx])
        etiqueta_final = grupo_a_fruta(best_idx)

        if score >= MIN_CONF:
            detecciones.append({
                "fruta": etiqueta_final,
                "score": score,
                "modelo": model_name,
                "mensaje": pred.get("mensaje_alerta", "")
            })

    # Mostrar imagen analizada
    st.image(img, caption="Imagen analizada", use_container_width=True)

    if not detecciones:
        st.subheader("⚠️ No se detectó fruta con suficiente confianza")
    else:
        detecciones = sorted(detecciones, key=lambda x: x["score"], reverse=True)
        top = detecciones[0]
        pct = int(top["score"] * 100)
        color = "green" if pct > 80 else "orange" if pct > 50 else "red"

        st.subheader(f"🍀 Fruta detectada: {top['fruta']}")
        st.markdown(f"**Confianza:** <span style='color:{color}'>{pct}%</span>", unsafe_allow_html=True)

        if top.get("mensaje"):
            st.info(top["mensaje"])

        if len(detecciones) > 1:
            st.markdown("**Otras detecciones:**")
            for d in detecciones[1:]:
                p = int(d["score"] * 100)
                st.write(f"- {d['fruta']} — {p}% (modelo: {d['modelo']})")

        st.markdown("---")
        st.subheader("📊 Confianzas por todos los modelos")
        for entry in classifiers:
            clf = entry["clf"]
            model_name = entry["name"]
            try:
                pred = clf.predict(img, etiquetas=None, guardar_csv=False, umbral=0.0)
                raw_out = pred.get("raw_output")
                if raw_out is None:
                    raw_out = np.array([pred.get("confianza", 0.0)])
                raw_out = np.asarray(raw_out).flatten()
                best_idx = int(np.argmax(raw_out))
                score = float(raw_out[best_idx])
                etiqueta_final = grupo_a_fruta(best_idx)
                st.write(f"{etiqueta_final} → {score:.3f}")
            except Exception as e:
                st.write(f"{model_name} → error: {e}")