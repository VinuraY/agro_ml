# ═══════════════════════════════════════════════════════════════
#  Convert trained model → TFLite (FLOAT32 + INT8) → C header for ESP32-S3
#
#  Fixes vs. the original script:
#   1. UnicodeEncodeError on Windows -- header file is now written with
#      explicit encoding='utf-8' instead of the platform default (cp1252).
#   2. "STEP 0" sanity check: runs the ORIGINAL Keras model (model.predict,
#      no TFLite at all) on the same raw images before any conversion. If
#      that's already broken, the bug is upstream of this script -- in the
#      training preprocessing or class-label order -- not in quantization.
#   3. Class index -> name mapping now prefers a class_indices.json saved
#      at training time, instead of guessing from sorted(os.listdir(test_dir)).
#      A wrong guess here produces exactly the symptom of "model is broken"
#      (near-zero or below-random accuracy) even when the model is fine.
# ═══════════════════════════════════════════════════════════════

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# ── Configuration ────────────────────────────────────────────
DATASET_DIR            = 'data/test'   # held-out test images, one subfolder per class
OUTPUT_DIR              = 'output'
IMG_HEIGHT              = 96
IMG_WIDTH               = 96
SAMPLES_PER_CLASS       = 20            # images per class used for accuracy checks
CALIBRATION_PER_CLASS   = 40            # images per class offered to the INT8 calibrator
CALIBRATION_MAX_TOTAL   = 200

# Two supported formats -- the script checks for both, in this priority order:
#   CLASS_NAMES_LIST_FILE: an ORDERED LIST, e.g. json.dump(train_ds.class_names, f)
#       -> ["Tomato___Bacterial_spot", "Tomato___Early_blight", ...]  index = position
#   CLASS_INDEX_DICT_FILE: a NAME -> INDEX MAP, e.g. json.dump(generator.class_indices, f)
#       -> {"Tomato___Bacterial_spot": 0, "Tomato___Early_blight": 1, ...}
CLASS_NAMES_LIST_FILE = f"{OUTPUT_DIR}/class_names.json"
CLASS_INDEX_DICT_FILE = f"{OUTPUT_DIR}/class_indices.json"

os.makedirs(f"{OUTPUT_DIR}/tflite", exist_ok=True)


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def load_class_names(dataset_dir, list_file, dict_file):
    """Resolve class index -> class name, preferring whatever was saved at
    training time. Falling back to alphabetical sort of the test folder is
    only correct if training assigned indices the same way -- which is common
    (it's the Keras default) but not guaranteed."""
    if os.path.exists(list_file):
        with open(list_file, 'r', encoding='utf-8') as f:
            names = json.load(f)  # ["class_name", ...] -- index = position
        print(f"Loaded class order from {list_file} (ordered list): {names}")
        return names

    if os.path.exists(dict_file):
        with open(dict_file, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)  # {"class_name": index, ...}
        names = [None] * len(class_indices)
        for name, idx in class_indices.items():
            names[idx] = name
        print(f"Loaded class order from {dict_file} (name->index map): {names}")
        return names

    print(f"WARNING: no {list_file} or {dict_file} found.")
    print("  Falling back to sorted(os.listdir(DATASET_DIR)). This is ONLY correct")
    print("  if your training pipeline also assigned indices alphabetically over")
    print("  these exact class names. If accuracy below comes out near/below")
    print("  random chance, this is the first thing to check.")
    return sorted(d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)))


def iter_test_images(dataset_dir, class_names, samples_per_class):
    """Yield (image_path, class_idx) for up to samples_per_class images per class."""
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"WARNING: expected class folder '{class_path}' not found, skipping")
            continue
        imgs = sorted(
            list(Path(class_path).glob('*.jpg')) +
            list(Path(class_path).glob('*.jpeg')) +
            list(Path(class_path).glob('*.png'))
        )
        for img_path in imgs[-samples_per_class:]:
            yield img_path, class_idx


def load_raw_image(img_path, img_h, img_w):
    """Load an image as raw 0-255 float32, shape (1, h, w, 3).
    No /255 here -- this model has true_divide/subtract layers baked in,
    so it expects raw pixel values as input. That's only correct if the
    training pipeline ALSO fed this exact model raw 0-255 pixels."""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_h, img_w))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(arr, 0).astype(np.float32)


def evaluate(predict_fn, dataset_dir, class_names, samples_per_class, label):
    """Run predict_fn(img_array) -> per-class scores over the test set and
    report overall + per-class accuracy, plus average top-1 confidence
    (a flat ~1/num_classes confidence suggests garbage/mismatched input;
    high confidence but wrong answers suggests a label/class-order bug)."""
    correct_per_class = {name: 0 for name in class_names}
    total_per_class = {name: 0 for name in class_names}
    confidences = []

    for img_path, class_idx in iter_test_images(dataset_dir, class_names, samples_per_class):
        img_array = load_raw_image(img_path, IMG_HEIGHT, IMG_WIDTH)
        output = predict_fn(img_array)
        predicted = int(np.argmax(output))
        confidences.append(float(np.max(output)))

        name = class_names[class_idx]
        total_per_class[name] += 1
        if predicted == class_idx:
            correct_per_class[name] += 1

    total_correct = sum(correct_per_class.values())
    total_count = sum(total_per_class.values())
    accuracy = total_correct / total_count if total_count else 0.0

    print(f"\n{label} accuracy: {accuracy*100:.2f}%  ({total_correct}/{total_count})")
    if confidences:
        chance = 100.0 / len(class_names)
        print(f"{label} avg top-1 confidence: {np.mean(confidences)*100:.1f}% "
              f"(uniform/random chance ~{chance:.1f}%)")
    for name in class_names:
        t = total_per_class[name]
        c = correct_per_class[name]
        if t:
            print(f"    {name:25s} {c}/{t}  ({c/t*100:.1f}%)")

    return accuracy


# ════════════════════════════════════════════════════════════════
#  LOAD MODEL + CLASS NAMES
# ════════════════════════════════════════════════════════════════

class_names = load_class_names(DATASET_DIR, CLASS_NAMES_LIST_FILE, CLASS_INDEX_DICT_FILE)
print(f"Classes ({len(class_names)}): {class_names}\n")

print("Loading trained model...")
model = tf.keras.models.load_model(f"{OUTPUT_DIR}/saved_model/plant_disease_model.keras")
print("Model loaded successfully\n")

print("=== MODEL ARCHITECTURE ===")
model.summary()
print("==========================\n")


# ════════════════════════════════════════════════════════════════
#  Sanity-check the ORIGINAL Keras model (no TFLite at all)
# ════════════════════════════════════════════════════════════════

print("=" * 60)
print("STEP 0: Sanity-checking the original Keras model (no TFLite)")
print("=" * 60)


def keras_predict(img_array):
    return model.predict(img_array, verbose=0)[0]


keras_accuracy = evaluate(keras_predict, DATASET_DIR, class_names, SAMPLES_PER_CLASS, "Original Keras model")

if keras_accuracy < (1.5 / len(class_names)):
    print("\n*** WARNING: the ORIGINAL Keras model -- no TFLite, no quantization --")
    print("*** is at/below chance level on this data. Nothing below this point")
    print("*** can fix that. Before chasing conversion bugs, check:")
    print("***   1) class_indices.json matches the order used at training time")
    print("***   2) the preprocessing here (raw 0-255 pixels) matches what the")
    print("***      training pipeline fed into THIS model")
    print("***   3) DATASET_DIR contains correctly labeled, held-out images")
    print()


# ════════════════════════════════════════════════════════════════
#  FLOAT32 TFLite (baseline, no quantization)
# ════════════════════════════════════════════════════════════════

print("\nConverting to FLOAT32 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float = converter.convert()

float_path = f"{OUTPUT_DIR}/tflite/model_float32.tflite"
with open(float_path, 'wb') as f:
    f.write(tflite_float)
print(f"  FLOAT32 size: {len(tflite_float)/1024:.1f} KB")

interpreter_float = tf.lite.Interpreter(model_content=tflite_float)
interpreter_float.allocate_tensors()
in_det_float = interpreter_float.get_input_details()
out_det_float = interpreter_float.get_output_details()


def float_predict(img_array):
    interpreter_float.set_tensor(in_det_float[0]['index'], img_array)
    interpreter_float.invoke()
    return interpreter_float.get_tensor(out_det_float[0]['index'])[0]


float_accuracy = evaluate(float_predict, DATASET_DIR, class_names, SAMPLES_PER_CLASS, "FLOAT32 TFLite")


# ════════════════════════════════════════════════════════════════
#  INT8 Quantized TFLite (for ESP32-S3)
# ════════════════════════════════════════════════════════════════

def representative_dataset_gen():
    image_paths = []
    for class_name in class_names:
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        imgs = sorted(
            list(Path(class_path).glob('*.jpg')) +
            list(Path(class_path).glob('*.png')) +
            list(Path(class_path).glob('*.jpeg'))
        )
        image_paths.extend(imgs[:CALIBRATION_PER_CLASS])

    np.random.shuffle(image_paths)
    image_paths = image_paths[:CALIBRATION_MAX_TOTAL]
    print(f"  Calibration images: {len(image_paths)}")

    for img_path in image_paths:
        yield [load_raw_image(img_path, IMG_HEIGHT, IMG_WIDTH)]


print("\nConverting to INT8 quantized TFLite...")
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset_gen
converter_int8.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_int8 = converter_int8.convert()

int8_path = f"{OUTPUT_DIR}/tflite/model_int8.tflite"
with open(int8_path, 'wb') as f:
    f.write(tflite_int8)

print(f"  INT8 size: {len(tflite_int8)/1024:.1f} KB")
print(f"  Size reduction: {len(tflite_float)/len(tflite_int8):.1f}x smaller")


# ════════════════════════════════════════════════════════════════
#  VALIDATE INT8 ACCURACY
# ════════════════════════════════════════════════════════════════

print("\nValidating INT8 model accuracy...")

interpreter = tf.lite.Interpreter(model_content=tflite_int8)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]['quantization']
out_scale, out_zero_point = output_details[0]['quantization']

print(f"  Input:  scale={input_scale:.6f}, zero_point={input_zero_point}")
print(f"  Output: scale={out_scale:.6f},   zero_point={out_zero_point}")


def int8_predict(img_array):
    img_int8 = img_array / input_scale + input_zero_point
    img_int8 = np.clip(img_int8, -128, 127).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], img_int8)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return (output.astype(np.float32) - out_zero_point) * out_scale


int8_accuracy = evaluate(int8_predict, DATASET_DIR, class_names, SAMPLES_PER_CLASS, "INT8 TFLite")


# ════════════════════════════════════════════════════════════════
#  GENERATE C HEADER FILE FOR ESP32-S3
# ════════════════════════════════════════════════════════════════

print("\nGenerating C header file for ESP32-S3...")


def generate_c_header(tflite_data, class_names, input_scale, input_zero_point,
                       out_scale, out_zero_point, img_h, img_w, accuracy):
    lines = []
    lines.append("// ═══════════════════════════════════════════════════")
    lines.append("// HydroNet Plant Disease Model — Auto-Generated")
    lines.append(f"// Input:    {img_h}x{img_w}x3 (INT8 quantized)")
    lines.append(f"// Classes:  {len(class_names)}")
    lines.append(f"// Accuracy: {accuracy*100:.2f}%")
    lines.append(f"// Size:     {len(tflite_data)/1024:.1f} KB")
    lines.append("// ═══════════════════════════════════════════════════")
    lines.append("")
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"#define MODEL_INPUT_HEIGHT    {img_h}")
    lines.append(f"#define MODEL_INPUT_WIDTH     {img_w}")
    lines.append("#define MODEL_INPUT_CHANNELS  3")
    lines.append(f"#define MODEL_NUM_CLASSES     {len(class_names)}")
    lines.append("")
    lines.append(f"#define MODEL_INPUT_SCALE      {input_scale:.8f}f")
    lines.append(f"#define MODEL_INPUT_ZERO_POINT {input_zero_point}")
    lines.append(f"#define MODEL_OUTPUT_SCALE     {out_scale:.8f}f")
    lines.append(f"#define MODEL_OUTPUT_ZERO_POINT {out_zero_point}")
    lines.append("")
    lines.append("// Class labels — index matches model output index")
    lines.append(f"const char* CLASS_NAMES[{len(class_names)}] = {{")
    for name in class_names:
        lines.append(f'  "{name}",')
    lines.append("};")
    lines.append("")
    lines.append(f"// Model weights — {len(tflite_data)} bytes")
    lines.append(f"const unsigned int MODEL_DATA_LEN = {len(tflite_data)};")
    lines.append("alignas(8) const unsigned char MODEL_DATA[] = {")

    hex_values = [f"0x{b:02x}" for b in tflite_data]
    for i in range(0, len(hex_values), 16):
        chunk = hex_values[i:i + 16]
        line = "  " + ", ".join(chunk)
        if i + 16 < len(hex_values):
            line += ","
        lines.append(line)

    lines.append("};")
    lines.append("")

    return "\n".join(lines)


header_content = generate_c_header(
    tflite_data=tflite_int8,
    class_names=class_names,
    input_scale=input_scale,
    input_zero_point=input_zero_point,
    out_scale=out_scale,
    out_zero_point=out_zero_point,
    img_h=IMG_HEIGHT,
    img_w=IMG_WIDTH,
    accuracy=int8_accuracy,
)

header_path = f"{OUTPUT_DIR}/model.h"

with open(header_path, 'w', encoding='utf-8') as f:
    f.write(header_content)

print(f"  Header saved: {header_path}")

print(f"\n{'═'*50}")
print("  CONVERSION COMPLETE")
print(f"{'═'*50}")
print(f"  Original Keras acc: {keras_accuracy*100:.2f}%")
print(f"  FLOAT32 TFLite acc: {float_accuracy*100:.2f}%")
print(f"  INT8 TFLite acc:    {int8_accuracy*100:.2f}%")
print(f"  FLOAT32 size:       {len(tflite_float)/1024:.1f} KB")
print(f"  INT8 size:          {len(tflite_int8)/1024:.1f} KB")
print(f"  Classes:            {class_names}")
print(f"\n  Files generated:")
print(f"  → {OUTPUT_DIR}/tflite/model_float32.tflite")
print(f"  → {OUTPUT_DIR}/tflite/model_int8.tflite")
print(f"  → {OUTPUT_DIR}/model.h  ← copy this to ESP32 src/")
print(f"{'═'*50}\n")