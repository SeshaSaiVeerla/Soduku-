import numpy as np
import cv2
from tensorflow.keras.models import load_model
from Create_cnn_model import create_cnn_model
from utils import preprocess_image, extract_digits

# Backtracking algorithm to solve Sudoku
def is_valid_move(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num or grid[x][col] == num:
            return False
        # Check the 3x3 subgrid
        if grid[row // 3 * 3 + x // 3][col // 3 * 3 + x % 3] == num:
            return False
    return True

def solve_sudoku(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                for num in range(1, 10):
                    if is_valid_move(grid, row, col, num):
                        grid[row][col] = num
                        if solve_sudoku(grid):
                            return True
                        grid[row][col] = 0
                return False
    return True

# Load or create CNN model
def load_or_create_model():
    try:
        model = load_model('digit_model.h5')  # Attempt to load pre-trained model
        print("Model loaded successfully.")
    except:
        print("Model not found. Creating a new model.")
        model = create_cnn_model()
        model.save('digit_model.h5')  # Save the model for later use
    return model

# Main function to run the solver
def process_image_input(image_path):
    grid_image = preprocess_image(image_path)
    model = load_or_create_model()
    
    sudoku_grid = extract_digits(grid_image, model)
    print("Recognized Sudoku Grid:")
    print(sudoku_grid)

    if solve_sudoku(sudoku_grid):
        print("Solved Sudoku Grid:")
        print(sudoku_grid)
    else:
        print("No solution found.")

if __name__ == "__main__":
    image_path = input("Enter the path to the Sudoku image: ")
    process_image_input(image_path)
