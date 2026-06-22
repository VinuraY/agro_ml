# This script tests the current trained model on the test dataset.
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration options.
IMG_SIZE = (96, 96)
BATCH_SIZE = 32

# Load the trained model.
model = tf.keras.models.load_model('plant_disease_model.keras')
print('Trained model loaded successfully.')


# Used to test entire dataset.
def batch_data_testing():

    # Load the test dataset.
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        'data/test',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    print('Test dataset loaded successfully.')

    class_names = test_dataset.class_names

    # Apply prefetch for performance
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    # =============================================================================
    # Collect predictions AND true labels in ONE single pass
    # =============================================================================
    print('Running predictions...')
    all_predictions = []
    all_true_labels = []

    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        all_predictions.extend(np.argmax(preds, axis=1).tolist())
        all_true_labels.extend(labels.numpy().tolist())

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # Sanity check
    print(f'Total True Labels : {len(all_true_labels)}')
    print(f'Total Predictions : {len(all_predictions)}')

    # =============================================================================
    # Classification Report
    # =============================================================================
    print('\n' + '='*50)
    print('CLASSIFICATION REPORT')
    print('='*50)
    print(classification_report(all_true_labels,
          all_predictions, target_names=class_names))

    # =============================================================================
    # Confusion Matrix
    # =============================================================================
    cm = confusion_matrix(all_true_labels, all_predictions)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Confusion matrix saved.')

    # =============================================================================
    # Overall accuracy
    # =============================================================================
    correct = np.sum(all_predictions == all_true_labels)
    total = len(all_true_labels)
    print(
        f'\n  Test Accuracy : {correct/total*100:.2f}%  ({correct}/{total} correct)')


# Used to test model prediction for a single image file.
def single_data_testing(image_path):

    # Load class names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)

    print(f'Loading : {image_path}')

    # Handles normalization part.
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)

    # Add batch dimension (1, 96, 96, 3)
    image = tf.expand_dims(image, axis=0)

    # Predict — preprocessing happens automatically inside the model
    predictions = model.predict(image, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100

    print(f'\n{"="*45}')
    print(f'  Image     : {image_path.split("/")[-1]}')
    print(f'  Diagnosis : {class_names[predicted_idx]}')
    print(f'  Confidence: {confidence:.2f}%')
    print(f'{"="*45}')

    print('\nAll class probabilities:')
    sorted_preds = sorted(enumerate(predictions[0]),
                          key=lambda x: x[1], reverse=True)
    
    for idx, prob in sorted_preds:
        bar = '█' * int(prob * 40)
        marker = ' ← predicted' if idx == predicted_idx else ''
        print(f'  {class_names[idx]:45s} {prob*100:6.2f}%  {bar}{marker}')

    return class_names[predicted_idx], confidence
