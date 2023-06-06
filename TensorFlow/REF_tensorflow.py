import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Activation
from tensorflow.keras.applications import ResNet50


data = pd.read_csv("fer2013.csv")

emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))

train_data = data[data['Usage']=='Training'] 
val_data = data[data['Usage']=='PublicTest'] 
test_data = data[data['Usage']=='PrivateTest']

X_train, y_train = np.array(list(train_data['pixels'])), train_data['emotion'].values
X_val, y_val = np.array(list(val_data['pixels'])), val_data['emotion'].values
X_test, y_test = np.array(list(test_data['pixels'])), test_data['emotion'].values


X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)


# Modification de l'architecture du modèle

model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(48,48,1)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),
    Flatten(),
    Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')

])


data_generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)




data_generator.fit(X_train)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Ajout d'un callback ReduceLROnPlateau pour réduire le taux d'apprentissage si la validation_loss ne s'améliore pas

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)




# Augmentation du nombre d'époques

epochs = 1
history = model.fit(data_generator.flow(X_train, y_train, batch_size=64), validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stopping, reduce_lr])




#Evaluer le modele

model.evaluate(X_test, y_test)

# Sauvegarder le modèle

model.save('emotion_recognition_model.h5')




plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# Faire des prédictions avec le modèle sur l'ensemble de test

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculer les métriques

print(classification_report(y_true_classes, y_pred_classes, target_names=list(emotion_map.values())))

# Afficher la matrice de confusion

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=list(emotion_map.values()), yticklabels=list(emotion_map.values()))
plt.title('Matrice de confusion')
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
plt.show()

############################   CHARGER LE MODELE + IMAGE AFIN D'ETABLIR UNE PREDICTION   #######################

# Charger le modèle sauvegardé

loaded_model = load_model('emotion_recognition_model.h5')


# Charger une image et effectuer les prétraitements nécessaires

img_path = 'sad.jpg'
img = image.load_img(img_path, grayscale=True, target_size=(48, 48))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255

# Faire une prédiction avec le modèle chargé

prediction = loaded_model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)

# Afficher la prédiction

print("Prédiction de l'émotion :", emotion_map[predicted_class[0]])