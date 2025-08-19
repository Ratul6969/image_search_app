# setup.py
import os
import sys
import json
import time 
from tqdm import tqdm 
import requests

# Add the project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our modular components
from config import (
    IMAGE_DIR, DATA_DIR, ANNOY_INDEX_PATH, HANDLE_PATH, DIMENSION_PATH,
    CLEANED_PRODUCTS_JSON_PATH, PRODUCTS_JSON_PATH
)
from utils.data_loader import load_products, load_cleaned_products
from models.efficientnet_extractor import EfficientNetFeatureExtractor
from models.vector_db import VectorDB

def download_image(url, save_path):
    """Downloads an image from a URL and saves it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def run_setup():
    print("🚀 Starting Image Search Project Setup...")

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n--- Phase 1: Checking and Downloading Product Images ---")
    
    try:
        with open(PRODUCTS_JSON_PATH, "r", encoding="utf-8") as f:
            products_raw = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Raw products JSON file not found at {PRODUCTS_JSON_PATH}")
        sys.exit(1)

    for product in tqdm(products_raw, desc="Downloading images"):
        image_url = product.get("Image Src")
        handle = product.get("Handle")
        if image_url and handle:
            image_path = os.path.join(IMAGE_DIR, f"{handle}.jpg")
            if not os.path.exists(image_path):
                download_image(image_url, image_path)

    if not os.path.exists(CLEANED_PRODUCTS_JSON_PATH):
        print(f"❌ Error: Cleaned products JSON file not found at {CLEANED_PRODUCTS_JSON_PATH}")
        print("Please run `python clean_product_data.py` first to prepare your data.")
        sys.exit(1)

    print("\n--- Phase 2: Loading Cleaned Product Data ---")
    try:
        products_for_indexing = load_cleaned_products(CLEANED_PRODUCTS_JSON_PATH) 
        
        if not products_for_indexing:
            print("No valid products found in cleaned JSON. Check `products_cleaned.json` and download logs.")
            sys.exit(1)
        print(f"Successfully prepared {len(products_for_indexing)} products for indexing.")

    except FileNotFoundError as e:
        print(f"❌ Setup Error: {e}.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("❌ Setup Error: Invalid JSON format in products_cleaned.json. Please check the file.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred during product data loading: {e}")
        sys.exit(1)

    print("\n--- Phase 3: Extracting Features and Building Vector Index ---")
    try:
        extractor = EfficientNetFeatureExtractor()
        actual_feature_dim = extractor.feature_dim 
        
        with open(DIMENSION_PATH, 'w') as f:
            f.write(str(actual_feature_dim))

        vector_db = VectorDB()

        features = []
        handles = []
        
        for product in tqdm(products_for_indexing, desc="Extracting features"):
            # Correctly access the product ID using the key from the cleaned data
            product_id = product['product_id']
            image_path = os.path.join(IMAGE_DIR, f"{product_id}.jpg")
            
            try:
                feature = extractor.get_features(image_path)
                features.append(feature)
                handles.append(product_id)
            except Exception as e:
                print(f"\nWarning: Could not extract features for {product_id} ({image_path}): {e}")
                continue 

        if not features:
            print("❌ No features extracted. Index cannot be built.")
            sys.exit(1)

        print(f"Extracted features for {len(features)} products.")
        vector_db.build_index(features, handles, actual_feature_dim) 
        print("✅ Indexing complete!")

    except Exception as e:
        print(f"❌ An error occurred during feature extraction or indexing: {e}")
        sys.exit(1)
        
    print("\n--- Setup Complete! You can now start the Flask app. ---")

if __name__ == '__main__':
    run_setup()
