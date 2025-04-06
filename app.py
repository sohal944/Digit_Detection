import gradio as gr
import numpy as np
import cv2
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("mnist_model.h5")

# Function to process image and predict
def predict_digit(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))

    # Invert image if needed (white background)
    if np.mean(resized) > 127:
        resized = 255 - resized

    # Normalize and flatten
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 784)

    # Predict
    prediction = model.predict(reshaped)
    return f"Predicted Digit: {np.argmax(prediction)}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="numpy", label="Upload a digit image"),
    outputs="text",
    title="MNIST Digit Recognizer",
    description="Upload a clear image of a handwritten digit (0-9) for prediction."
)

# Run the app
if __name__ == "__main__":
    interface.launch()
