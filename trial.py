import numpy as np
import cv2
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("mnist_model.h5")

# Load the input image (path to your jpg)
img = cv2.imread("5photo.jpg", cv2.IMREAD_GRAYSCALE)

# Resize to 28x28 (MNIST standard)
img = cv2.resize(img, (28, 28))

# Invert colors if necessary: MNIST expects white (255) digit on black (0) bg
img = 255 - img  

# Normalize pixel values
img = img / 255.0

# Flatten and reshape for model input
input_data = img.reshape(1, 28 * 28)

# Predict
prediction = model.predict(input_data)
predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)

# Optional: Show input image
cv2.imshow("Input", cv2.resize(img * 255, (200, 200)))
cv2.waitKey(0)
cv2.destroyAllWindows()
