
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix

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
#model = tf.keras.models.load_model('flower_classifier50_weights.h5')    #epoch 25
# model = tf.keras.models.load_model('flower_classifier3new_weights.h5')  #35

#model = tf.keras.models.load_model('flower_classifierallold_weights.h5') #25all
#model = tf.keras.models.load_model('flower_classifierallnew_weights.h5')    #35 all
#model = tf.keras.models.load_model('flower_classifierallnewlayer_weights.h5')    #64+ all
#model = tf.keras.models.load_model('flower_classifierallnewlayer5_weights.h5')    #64+ 32all
model = tf.keras.models.load_model('saved_models/flower_classifierallnewlayer6_weights.h5')    #64+ 64+32all

# Path to your test image
test_image_path = 'data/test/Image_45.jpg'
base_dir = 'data/train/'  # The directory containing flower class folders
class_names = sorted(os.listdir(base_dir))  # This assumes folder names match class names
print(class_names)
# if '.DS_Store' in class_names:
#     class_names.remove('dstore')
class_names = [class_name for class_name in class_names if class_name != '.DS_Store']

# # Predict the class
# predicted_class, confidence = predict_flower_class(test_image_path, model, class_names)

# print(f"Predicted class: {predicted_class}")
# print(f"Confidence: {confidence:.2%}")

# # You can also visualize the image with its prediction
# plt.figure(figsize=(6, 6))
# img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
# plt.imshow(img)
# plt.title(f"Predicted: {predicted_class} ({confidence:.2%})")
# plt.axis('off')
# plt.show()
test_folder = 'data/test/'
# List of specific image filenames

# #sunflowers
selected_filenames = [
    'Image_1.jpg', 'Image_2.jpg','Image_9.jpg',
    'Image_700.jpg',
]
#rose
selected_filenames = [
    'Image_4.jpg', 'Image_5.jpg','Image_22.jpg',
    'Image_535.jpg',
]
#dandelion
selected_filenames = [
    'Image_7.jpg', 'Image_465.jpg','Image_419.jpg','Image_351.jpg',
]
# tulip
selected_filenames = [
    'Image_70.jpg', 'Image_157.jpg','Image_271.jpg','Image_300.jpg',
]
# # tdaisy
selected_filenames = [
    'Image_102.jpg', 'Image_409.jpg','Image_408.jpg','Image_588.jpg',
]


# Create full paths for selected files
test_image_paths = [os.path.join(test_folder, fname) for fname in selected_filenames]



# # Gather all image paths in test folder
# test_image_paths = [os.path.join(test_folder, fname) 
#                     for fname in os.listdir(test_folder) 
#                     if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Plot setup

# test_image_paths = sorted(test_image_paths)[:20]  # Limit to first 20
n_images = len(test_image_paths)
print(n_images)
cols = 4
rows = (n_images + cols - 1) // cols
plt.figure(figsize=(15, 4 * rows))

# Predict and display each image
for i, image_path in enumerate(test_image_paths):
    predicted_class, confidence = predict_flower_class(image_path, model, class_names)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(f"{predicted_class} ({confidence:.1%})")
    plt.axis('off')

plt.tight_layout()
plt.show()
