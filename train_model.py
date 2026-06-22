# This script train the MobileNetV2 model on the dataset and save the trained model to a file.
# Only planned to modify classifier part of the model, not the feature extractor part.
import json
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model


# Configuration options.
IMG_SIZE = (96, 96) # Change the pixel values
BATCH_SIZE = 32
EPOCHS = 15
ALPHA = 0.5
AUTOTUNE = tf.data.AUTOTUNE

# Load the datasets.
train_dataset = keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = keras.utils.image_dataset_from_directory(
    'data/validation',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print('Datasets loaded successfully.')

# Get class names and save it to a json file.
class_names = train_dataset.class_names

with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

number_classes = len(class_names)

# Calculate Class Weights to fix the imbalance (804 samples vs 56 samples)
labels = np.concatenate([y for x, y in train_dataset], axis=0)
class_counts = np.bincount(labels)
total_samples = len(labels)
class_weights = {i: total_samples / (number_classes * count)
                 for i, count in enumerate(class_counts)}

# Auto-tune the datasets for performance.
train_dataset = train_dataset.cache().shuffle(
    1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

####################
#  Build the model #
####################

# Data augmentation layers to improve generalization.
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal_and_vertical'),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.1),
], name='data_augmentation')

input_shape = (*IMG_SIZE, 3)  # RGB images

base_model = MobileNetV2(input_shape=input_shape,
                         alpha=ALPHA, include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

inputs = keras.Input(shape=input_shape)

# Apply data augmentation.
x = data_augmentation(inputs)

# Preprocess the inputs for MobileNetV2.
x = keras.applications.mobilenet_v2.preprocess_input(x)

# Use the base model in inference mode.
x = base_model(x, training=False)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)

# Final classification layer with softmax activation for multi-class classification.
outputs = Dense(number_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()

# Compile the model with an appropriate optimizer and loss function for multi-class classification.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callback to save the best model during training.
callback = [
    keras.callbacks.ModelCheckpoint(
        'best_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)

]

#####################
#  Train the model  #
#####################

print('Starting training...')
history = model.fit(train_dataset, epochs=EPOCHS,
                    validation_data=validation_dataset, callbacks=callback)

# Save the trained model to a file.
model.save('plant_disease_model.keras')
print('Model saved to plant_disease_model.keras')


# Get model statistics.
def plot_training_history(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Training Analysis', fontsize=16, fontweight='bold')

    # ------------------------------------------------------------------
    # Plot 1 — Training vs Validation Accuracy
    # ------------------------------------------------------------------
    axes[0, 0].plot(epochs, acc,     'b-o',
                    label='Train Accuracy',      linewidth=2, markersize=4)
    axes[0, 0].plot(epochs, val_acc, 'r-o',
                    label='Validation Accuracy', linewidth=2, markersize=4)
    axes[0, 0].set_title('Training vs Validation Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Plot 2 — Training vs Validation Loss
    # ------------------------------------------------------------------
    axes[0, 1].plot(epochs, loss,     'b-o', label='Train Loss',
                    linewidth=2, markersize=4)
    axes[0, 1].plot(epochs, val_loss, 'r-o',
                    label='Validation Loss', linewidth=2, markersize=4)
    axes[0, 1].set_title('Training vs Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Plot 3 — Error Rate (1 - accuracy)
    # ------------------------------------------------------------------
    train_error = [1 - a for a in acc]
    val_error = [1 - a for a in val_acc]

    axes[1, 0].plot(epochs, train_error, 'b-o',
                    label='Train Error Rate',      linewidth=2, markersize=4)
    axes[1, 0].plot(epochs, val_error,   'r-o',
                    label='Validation Error Rate', linewidth=2, markersize=4)
    axes[1, 0].set_title('Error Rate over Epochs')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Error Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Plot 4 — Overfitting Gap (val_loss - train_loss)
    # ------------------------------------------------------------------
    gap = [v - l for v, l in zip(val_loss, loss)]

    axes[1, 1].plot(epochs, gap, 'g-o', linewidth=2, markersize=4)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].fill_between(epochs, gap, 0,
                            where=[g > 0 for g in gap],
                            alpha=0.2, color='red',   label='Overfitting zone')
    axes[1, 1].fill_between(epochs, gap, 0,
                            where=[g <= 0 for g in gap],
                            alpha=0.2, color='green', label='Underfitting zone')
    axes[1, 1].set_title('Overfitting Gap (val_loss − train_loss)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')

    # ------------------------------------------------------------------
    # Print final summary
    # ------------------------------------------------------------------
    print(f"\n{'='*45}")
    print(f"  Final Training Summary")
    print(f"{'='*45}")
    print(f"  Train     Accuracy : {acc[-1]:.4f}  ({acc[-1]*100:.2f}%)")
    print(
        f"  Val       Accuracy : {val_acc[-1]:.4f}  ({val_acc[-1]*100:.2f}%)")
    print(f"  Train     Loss     : {loss[-1]:.4f}")
    print(f"  Val       Loss     : {val_loss[-1]:.4f}")
    print(
        f"  Train     Error    : {train_error[-1]:.4f}  ({train_error[-1]*100:.2f}%)")
    print(
        f"  Val       Error    : {val_error[-1]:.4f}  ({val_error[-1]*100:.2f}%)")
    print(f"  Overfit   Gap      : {gap[-1]:.4f}")
    print(
        f"  Best Val  Accuracy : {max(val_acc):.4f} (Epoch {val_acc.index(max(val_acc)) + 1})")
    print(
        f"  Best Val  Loss     : {min(val_loss):.4f} (Epoch {val_loss.index(min(val_loss)) + 1})")
    print(f"{'='*45}")


# Call it after model.fit()
plot_training_history(history)
