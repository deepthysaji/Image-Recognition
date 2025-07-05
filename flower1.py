import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
# Path to your organized dataset
#base_dir = 'flowers3/'
base_dir = 'train/'  # The directory containing class_1, class_2, etc.

# Parameters
img_height, img_width = 224, 224  # Standard input size for many CNNs
batch_size = 32
epochs = 25
num_classes = len(os.listdir(base_dir))  # Number of flower classes

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)

# Only rescaling for validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create generators
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = val_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save class mapping for later use
class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
print(f"Classes: {class_names}")

# Use a pre-trained model as base
# base_model = MobileNetV2(
#     input_shape=(img_height, img_width, 3),
#     include_top=False,
#     weights='imagenet'
# )

# # Freeze the base model
# base_model.trainable = False

# # Create new model on top
# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(num_classes, activation='softmax')
# ])


# Define a simpler CNN architecture (no pre-trained model)
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),
    
    # First convolutional block
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional block
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # #Fourth convolutional block
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    #fifth convolutional block
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    #sixth convolutional block
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    
    # Classification head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])
# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()


# Add callbacks for early stopping and saving best model
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
#     tf.keras.callbacks.ModelCheckpoint('best_flower_model.h5', save_best_only=True)
# ]

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

model.save('flower_classifierallold_weights.h5')
print("Final model saved to 'flower_classifier_final.h5'")

# If you want to also save just the weights
model.save_weights('flower_classifier150.weights.h5')
print("Model weights saved to 'flower_classifier_weights.h5'")

def predict_flower_class(image_path, model, class_names):
    """
    Predicts the flower class of an image using the trained model.
    
    Parameters:
    - image_path: Path to the image file to classify
    - model: Trained Keras model
    - class_names: List of class names corresponding to model outputs
    
    Returns:
    - Predicted class name and confidence score
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to 0-1
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get predicted class index and confidence
    predicted_class_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_index])
    
    # Get class name
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name, confidence

# Example usage
# First, load your saved model
#model = tf.keras.models.load_model('best_flower_model.h5')

# Path to your test image
test_image_path = 'test/Image_1.jpg'

# Predict the class
predicted_class, confidence = predict_flower_class(test_image_path, model, class_names)

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")

# You can also visualize the image with its prediction
plt.figure(figsize=(6, 6))
img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
plt.imshow(img)
plt.title(f"Predicted: {predicted_class} ({confidence:.2%})")
plt.axis('off')
plt.show()