# config.py
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
IMAGE_DIR = os.path.join(DATA_DIR, 'product_images')

# File Paths
PRODUCTS_JSON_PATH = os.path.join(BASE_DIR, 'products.json')
CLEANED_PRODUCTS_JSON_PATH = os.path.join(BASE_DIR, 'products_cleaned.json')  # Root directory
DATA_PRODUCTS_CLEANED_PATH = os.path.join(DATA_DIR, 'products_cleaned.json')  # Data directory
ANNOY_INDEX_PATH = os.path.join(DATA_DIR, 'product_index.ann')
HANDLE_PATH = os.path.join(DATA_DIR, 'handles.npy')
DIMENSION_PATH = os.path.join(DATA_DIR, 'feature_dimension.txt')

# Gemini API Key - Use environment variable for security
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Model and search parameters
TOP_K_DISPLAY = 10
TOP_K_CANDIDATES = 50