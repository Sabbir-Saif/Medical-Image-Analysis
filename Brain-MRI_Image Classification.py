# Importing Libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense,Flatten,BatchNormalization,Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
# %matplotlib inline
import zipfile
import os
from PIL import Image
import io
import numpy as np
import pandas as pd

from tensorflow.keras.layers import MaxPool2D

''' Dataset is collected from Kaggle named brain-tumor-mri-dataset '''

path='D:\\AIP\\NLP\\archive (6).zip'
label_map = {
    "Training/glioma": 0,
    "Training/meningioma": 1,
    "Training/notumor": 2,
    "Training/pituitary": 3
}
dataset = []  # List to store (image, label) tuples

with zipfile.ZipFile(path, 'r') as zip_ref:
    file_list = zip_ref.namelist()  # Get all files inside the ZIP

    for folder, label in label_map.items():
        image_files = [f for f in file_list if f.startswith(folder) and f.endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in image_files:
            with zip_ref.open(img_file) as img_data:
                img = Image.open(io.BytesIO(img_data.read()))  # Load image

                img = img.convert("RGB")  # Convert all images to RGB (3 channels)

                img = img.resize((128, 128))  # Resize for consistency
                img_array = np.array(img)  # Convert to NumPy array
                dataset.append((img_array, label))  # Store image and label

print(f"Total images loaded: {len(dataset)}")

# Convert dataset to NumPy arrays
X = np.array([img for img, label in dataset])  # Image data
y = np.array([label for img, label in dataset])  # Labels

print("Dataset shape:", X.shape, "Labels shape:", y.shape)

X = np.array([img for img, label in dataset])  # Image data
y = np.array([label for img, label in dataset])  # Labels

print("Dataset shape:", X.shape, "Labels shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split 80% training, 20% temporary (for validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Split 50% of remaining into validation & test (10% each of original dataset)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print sizes
print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
print(f"Validation set: {X_val.shape}, Labels: {y_val.shape}")
print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")

#Step 2: Normalize and Preprocess Data for InceptionV3
#InceptionV3 requires input images preprocessed using its own function.

from tensorflow.keras.applications.inception_v3 import preprocess_input

X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)

#Scales pixel values to match InceptionV3 expectations

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=4)
y_val = to_categorical(y_val, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

base_model = InceptionV3(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# Freeze the first 271 layers
for layer in base_model.layers[:271]:
    layer.trainable = False

# Add custom layers
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=x)

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()  # Print model architecture

model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=64, epochs=5)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
from sklearn.metrics import classification_report

# Get model predictions on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_true = np.argmax(y_test, axis=1)  # Convert one-hot labels back to integers

# Print accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Generate classification report
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_true = np.argmax(y_test, axis=1)  # Convert one-hot labels back to integers

cm = confusion_matrix(y_true, y_pred_classes)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm, classes=['glioma', 'meningioma', 'notumor', 'pituitary'])

from tensorflow.keras.applications import VGG16

# Load the pretrained VGG16 model
base_model = VGG16(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)  # Dropout for regularization
x = layers.Dense(4, activation='softmax')(x)

# Define the new model
model = tf.keras.models.Model(base_model.input, x)

# Compile and train
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=5)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
from sklearn.metrics import classification_report

# Get model predictions on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_true = np.argmax(y_test, axis=1)  # Convert one-hot labels back to integers

# Print accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Generate classification report
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# Define ResNet50

from tensorflow.keras.applications import ResNet50

base_model = ResNet50(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification
x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)  # Dropout for regularization
x = layers.Dense(4, activation='softmax')(x)

# Define the new model
model = tf.keras.models.Model(base_model.input, x)

# Compile and train
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=5)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
from sklearn.metrics import classification_report

# Get model predictions on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_true = np.argmax(y_test, axis=1)  # Convert one-hot labels back to integers

# Print accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Generate classification report
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(classification_report(y_true, y_pred_classes, target_names=class_labels))
