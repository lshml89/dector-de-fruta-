# Clasificador de Frutas 🍌🍊

Este proyecto es una aplicación desarrollada con **Streamlit** que utiliza un modelo de aprendizaje no supervisado en formato **TensorFlow Lite (.tflite)** para detectar si una imagen corresponde a un **Plátano** o una **Naranja**.

## 🚀 ¿Cómo funciona?

- El usuario puede subir una imagen o usar la cámara.
- El modelo analiza la imagen y asigna un grupo.
- Cada grupo está mapeado manualmente:
  - Grupo 0 → Plátano 🍌
  - Grupo 1 → Naranja 🍊
- Se muestra la fruta detectada junto con el nivel de confianza.

## 📦 Estructura del proyecto


## 🧪 Requisitos

- Python 3.8+
- Streamlit
- TensorFlow
- Pillow
- NumPy

Instalar dependencias:

```bash
pip install -r requirements.txt

para ejecutar streamlit run app.py 
