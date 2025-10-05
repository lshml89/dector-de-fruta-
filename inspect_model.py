import glob, os, sys
import numpy as np
from PIL import Image
import tensorflow as tf

def load_image(path, size=(224, 224)):
    """Carga y preprocesa una imagen para el modelo."""
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        print(f"Error al cargar la imagen '{path}': {e}")
        return None

def inspect(model_path, labels_path=None, test_image=None):
    print("=== Modelo:", model_path)
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print(" input shape:", inp['shape'], "dtype:", inp.get('dtype'))
    print(" output shape:", out['shape'], "dtype:", out.get('dtype'))

    # Cargar etiquetas si existen
    labels = None
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [l.strip() for l in f if l.strip()]
        print(" labels:", labels[:5], "(total", len(labels), ")")
    else:
        print(" labels: NONE or file missing:", labels_path)

    # Ejecutar inferencia si se proporciona imagen
    if test_image:
        h = int(inp['shape'][1]) if len(inp['shape']) > 1 else 224
        w = int(inp['shape'][2]) if len(inp['shape']) > 2 else 224
        arr = load_image(test_image, size=(w, h))
        if arr is None:
            return

        dtype = inp.get('dtype', np.float32)
        try:
            arr_set = (arr * 255.0).astype(dtype) if np.issubdtype(np.dtype(dtype), np.integer) else arr.astype(dtype)
        except Exception:
            arr_set = arr.astype(np.float32)

        try:
            interpreter.set_tensor(inp['index'], arr_set)
            interpreter.invoke()
            out_data = interpreter.get_tensor(out['index'])
        except Exception as e:
            print(f"❌ Error durante la inferencia: {e}")
            return

        print(" raw output (shape):", np.shape(out_data))
        print(" sample raw values:", np.ravel(out_data)[:10])

        v = np.ravel(out_data)
        if v.size == 1:
            print(" single-output model -> value:", v[0])
        else:
            e = np.exp(v - np.max(v))
            probs = e / e.sum()
            top3 = sorted(list(enumerate(probs)), key=lambda x: -x[1])[:3]
            print(" softmax top3:")
            for idx, score in top3:
                label = labels[idx] if labels and idx < len(labels) else f"Class {idx}"
                print(f"  - {label}: {score:.3f}")

if __name__ == "__main__":
    folder = "modelos"
    models = sorted(glob.glob(os.path.join(folder, "*.tflite")))
    if not models:
        print("No models in", folder)
        sys.exit(1)

    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    for m in models:
        lbl = m.replace(".tflite", ".txt")
        inspect(m, labels_path=lbl, test_image=test_img)
        print()