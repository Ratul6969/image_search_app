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
from utils.data_loader import load_products 
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
        # Load both raw and cleaned data to link URLs and filenames
        products_raw = load_products(PRODUCTS_JSON_PATH)
        products_for_indexing = load_products(CLEANED_PRODUCTS_JSON_PATH) 
    except FileNotFoundError as e:
        print(f"❌ Error: One of the product JSON files was not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in one of the product files: {e}")
        sys.exit(1)

    if not products_for_indexing:
        print("No valid products found in cleaned JSON. Check `products_cleaned.json`.")
        sys.exit(1)
    
    # Create a mapping from product_id to image URL for efficient lookup
    url_map = {p.get('Handle'): p.get('Image Src') for p in products_raw if p.get('Handle')}

    # Download images using the correct filenames from products_cleaned.json
    for product in tqdm(products_for_indexing, desc="Downloading images"):
        product_id = product.get('product_id')
        image_file = product.get('image_file')

        if product_id and image_file:
            image_url = url_map.get(product_id)
            if image_url:
                image_path = os.path.join(IMAGE_DIR, image_file)
                if not os.path.exists(image_path):
                    download_image(image_url, image_path)
            else:
                print(f"\nWarning: Image URL not found for product_id: {product_id}. Skipping download.")


    print("\n--- Phase 2: Extracting Features and Building Vector Index ---")
    try:
        extractor = EfficientNetFeatureExtractor()
        actual_feature_dim = extractor.feature_dim 
        
        with open(DIMENSION_PATH, 'w') as f:
            f.write(str(actual_feature_dim))

        vector_db = VectorDB()

        features = []
        handles = []
        
        for product in tqdm(products_for_indexing, desc="Extracting features"):
            product_id = product['product_id']
            image_file = product['image_file']
            image_path = os.path.join(IMAGE_DIR, image_file)
            
            try:
                feature = extractor.get_features(image_path)
                features.append(feature)
                handles.append(product_id)
            except FileNotFoundError as fne:
                 print(f"\nWarning: Could not extract features for {product_id} as image was not found at {image_path}. Skipping product.")
                 continue
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