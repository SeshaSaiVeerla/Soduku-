import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

# Preprocess the Sudoku image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    return thresh

# Extract digits from the grid using the trained CNN model
def extract_digits(grid_image, model):
    digits = []
    grid_size = grid_image.shape[0] // 9  # Assuming a 9x9 grid of cells

    for i in range(9):
        row = []
        for j in range(9):
            # Extract each 28x28 cell
            cell = grid_image[i * grid_size:(i + 1) * grid_size, j * grid_size:(j + 1) * grid_size]
            cell_resized = cv2.resize(cell, (28, 28))  # Resize to 28x28 for the CNN

            # Preprocess for the model
            cell_tensor = image.img_to_array(cell_resized)
            cell_tensor = np.expand_dims(cell_tensor, axis=0)
            cell_tensor = cell_tensor / 255.0  # Normalize the image

            # Predict using the CNN model
            pred = model.predict(cell_tensor)
            digit = np.argmax(pred)  # Get the digit with the highest probability
            row.append(digit)
        digits.append(row)

    return np.array(digits)
