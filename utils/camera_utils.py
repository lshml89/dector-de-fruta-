import os
import csv
from typing import Optional, Union, Sequence

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


class Classifier:
    """
    Clasificador TFLite robusto para inferencia.
    - model_path: ruta al .tflite
    - csv_path: ruta para guardar historial (opcional)
    - umbral: confianza mínima por defecto (0..1)
    """
    def __init__(self, model_path: str, csv_path: str = "data/frutas.csv", umbral: float = 0.7):
        self.model_path = model_path
        self.csv_path = csv_path
        self.umbral = float(umbral)
        self.model_name = os.path.basename(model_path).replace(".tflite", "")

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            shape = self.input_details[0]['shape']
            self.input_shape = (int(shape[1]), int(shape[2])) if len(shape) >= 3 else (224, 224)

            if self.csv_path:
                os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
                if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                    with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(["fruta_predicha", "confianza"])
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el modelo TFLite '{model_path}': {e}")

    # -------------------------------
    # Métodos auxiliares
    # -------------------------------
    def _load_labels(self, etiquetas: Union[None, str, Sequence[str]]) -> Optional[list]:
        if etiquetas is None:
            return None
        if isinstance(etiquetas, str) and os.path.exists(etiquetas):
            with open(etiquetas, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return list(etiquetas) if isinstance(etiquetas, Sequence) else [etiquetas]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _is_prob_vector(self, v: np.ndarray) -> bool:
        return v.size > 0 and not np.any(np.isnan(v)) and not np.any(np.isinf(v)) and np.all(v >= 0) and abs(v.sum() - 1.0) < 1e-3

    def _label_fallback(self, raw_label: Optional[str]) -> str:
        if not raw_label or "class" in raw_label.lower() or raw_label.strip().isdigit():
            return self._modelname_to_friendly()
        return raw_label.strip()

    def _modelname_to_friendly(self) -> str:
        name = self.model_name.lower()
        if "naran" in name or "orange" in name:
            return "Naranja 🍊"
        if "platan" in name or "platano" in name or "banana" in name:
            return "Plátano 🍌"
        return self.model_name.replace("_", " ").title()

    def _is_label_readable(self, label: Optional[str]) -> bool:
        if not label:
            return False
        s = label.strip()
        if "class" in s.lower() or (len(s) <= 2 and any(ch.isdigit() for ch in s)):
            return False
        return any(ch.isalpha() for ch in s)

    # -------------------------------
    # Preprocesamiento de imagen
    # -------------------------------
    def preprocess(self, frame: Union[Image.Image, np.ndarray], normalize: bool = True, auto_rotate: bool = True) -> np.ndarray:
        if isinstance(frame, Image.Image):
            img = ImageOps.exif_transpose(frame.convert("RGB")) if auto_rotate else frame.convert("RGB")
            arr = np.asarray(img)
        else:
            arr = np.asarray(frame)

        try:
            pil = Image.fromarray(arr.astype('uint8'), 'RGB').resize(self.input_shape, Image.BILINEAR)
            tensor = np.asarray(pil)
        except Exception:
            tensor = tf.image.resize(tf.convert_to_tensor(arr), self.input_shape).numpy()

        if tensor.ndim == 3:
            tensor = np.expand_dims(tensor, axis=0)

        tensor = tensor.astype(np.float32) / 255.0 if normalize else tensor.astype(np.float32)

        expected_dtype = self.input_details[0].get('dtype', np.float32)
        try:
            tensor = tensor.astype(np.dtype(expected_dtype))
        except Exception:
            tensor = tensor.astype(np.float32)

        return tensor

    # -------------------------------
    # Predicción
    # -------------------------------
    def predict(self,
                frame: Union[Image.Image, np.ndarray],
                etiquetas: Union[None, str, Sequence[str]] = None,
                guardar_csv: bool = True,
                umbral: Optional[float] = None) -> dict:
        try:
            labels = self._load_labels(etiquetas)
            effective_umbral = float(self.umbral if umbral is None else umbral)
            input_data = self.preprocess(frame)

            input_index = self.input_details[0]['index']
            input_dtype = self.input_details[0].get('dtype', np.float32)
            input_np_dtype = np.dtype(input_dtype) if input_dtype else np.float32

            tensor_to_set = (input_data * 255.0).astype(input_np_dtype) if np.issubdtype(input_np_dtype, np.integer) else input_data.astype(input_np_dtype)

            self.interpreter.set_tensor(input_index, tensor_to_set)
            self.interpreter.invoke()

            output_index = self.output_details[0]['index']
            out = np.array(self.interpreter.get_tensor(output_index)).flatten()

            # Interpretar salida
            if out.size == 1:
                val = float(out[0])
                score = val / 255.0 if np.issubdtype(out.dtype, np.integer) else self._sigmoid(val)
                probs = np.array([score, 1.0 - score])
                best_idx = 0
                best_conf = score
            else:
                probs = self._softmax(out) if not self._is_prob_vector(out) else out
                best_idx = int(np.argmax(probs))
                best_conf = float(probs[best_idx])

            raw_label = labels[best_idx] if labels and best_idx < len(labels) else f"Class {best_idx}"
            display_label = raw_label if self._is_label_readable(raw_label) else self._label_fallback(raw_label)

            if best_conf < effective_umbral:
                resultado_csv = "Desconocido 🤔"
                mensaje = f"⚠️ Confianza baja ({best_conf*100:.1f}%)"
            else:
                resultado_csv = display_label
                mensaje = f"✅ Confianza aceptable ({best_conf*100:.1f}%)"

            if guardar_csv and self.csv_path:
                try:
                    with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([resultado_csv, round(best_conf, 3)])
                except Exception:
                    pass

            return {
                "resultado": resultado_csv,
                "confianza": best_conf,
                "probs": probs,
                "raw_output": out,
                "mensaje_alerta": mensaje
            }

        except Exception as e:
            raise RuntimeError(f"Error durante la predicción con el modelo '{self.model_path}': {e}")