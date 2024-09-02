import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# Etiketler
class_labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Veri kümeleri
images = []
labels = []
image_dim = 128

# Eğitim verisini yükle
for label in class_labels:
    folder_path = os.path.join('archive/', 'Training/', label)
    for file_name in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        try:
            image = Image.open(file_path)
            image = image.resize((image_dim, image_dim))
            images.append(np.array(image))
            labels.append(label)
        except:
            continue

# Test verisini yükle
for label in class_labels:
    folder_path = os.path.join('archive/', 'Testing/', label)
    for file_name in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        try:
            image = Image.open(file_path)
            image = image.resize((image_dim, image_dim))
            images.append(np.array(image))
            labels.append(label)
        except:
            continue

images = np.array(images)
labels = np.array(labels)

# Veriyi karıştır
images, labels = shuffle(images, labels, random_state=101)

# Veri artırma
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
datagen.fit(images)

# Eğitim ve test verisine ayır
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.1, random_state=101)

# Etiketleri sayısal forma çevir
labels_train_numeric = [class_labels.index(label) for label in labels_train]
labels_train = tf.keras.utils.to_categorical(labels_train_numeric)

labels_test_numeric = [class_labels.index(label) for label in labels_test]
labels_test = tf.keras.utils.to_categorical(labels_test_numeric)

# Model oluşturma
base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(image_dim, image_dim, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(rate=0.5)(x)
x = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

# Model özeti
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Callbacks
tensorboard_callback = TensorBoard(log_dir='logs')
checkpoint_callback = ModelCheckpoint("resnet152v1.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_accuracy', factor=0.4, patience=2, min_delta=0.001, mode='auto', verbose=1)

# Modeli eğit
history = model.fit(images_train, labels_train, validation_split=0.1, epochs=5, verbose=1, batch_size=64,
                    callbacks=[tensorboard_callback, checkpoint_callback, reduce_lr_callback])

# Eğitim sürecini görselleştir
plt.figure(figsize=(14, 7))
epochs_range = range(5)
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
train_accuracy = history.history['accuracy']
train_loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

# Renk paletleri
dark_colors = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
red_colors = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
green_colors = ['#01411C', '#4B6F44', '#4F7942', '#74C365', '#D0F0C0']

sns.palplot(dark_colors)
sns.palplot(green_colors)
sns.palplot(red_colors)

fig.text(s='Epochs vs. Training and Validation Accuracy/Loss', size=18, fontweight='bold',
         fontname='monospace', color=dark_colors[1], y=1, x=0.28, alpha=0.8)

sns.despine()
ax[0].plot(epochs_range, train_accuracy, marker='o', markerfacecolor=green_colors[2], color=green_colors[3],
           label='Training Accuracy')
ax[0].plot(epochs_range, val_accuracy, marker='o', markerfacecolor=red_colors[2], color=red_colors[3],
           label='Validation Accuracy')
ax[0].legend(frameon=False)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')

sns.despine()
ax[1].plot(epochs_range, train_loss, marker='o', markerfacecolor=green_colors[2], color=green_colors[3],
           label='Training Loss')
ax[1].plot(epochs_range, val_loss, marker='o', markerfacecolor=red_colors[2], color=red_colors[3],
           label='Validation Loss')
ax[1].legend(frameon=False)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')

plt.show()

# Model tahmini ve değerlendirme
predictions = model.predict(images_test)
predictions = np.argmax(predictions, axis=1)
labels_test_numeric = np.argmax(labels_test, axis=1)

print(classification_report(labels_test_numeric, predictions))

# Karışıklık matrisini görselleştir
fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(confusion_matrix(labels_test_numeric, predictions), ax=ax, xticklabels=class_labels, yticklabels=class_labels, annot=True,
            cmap=green_colors[::-1], alpha=0.7, linewidths=2, linecolor=dark_colors[3])
fig.text(s='Heatmap of the Confusion Matrix', size=18, fontweight='bold',
         fontname='monospace', color=dark_colors[1], y=0.92, x=0.28, alpha=0.8)
plt.show()
