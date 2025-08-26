# setup.py - More robust version with better debugging
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
    CLEANED_PRODUCTS_JSON_PATH, DATA_PRODUCTS_CLEANED_PATH
)
from utils.data_loader import load_products 
from models.efficientnet_extractor import EfficientNetFeatureExtractor
from models.vector_db import VectorDB

def download_image(url, save_path):
    """Downloads an image from a URL and saves it locally."""
    try:
        response = requests.get(url, stream=True, timeout=30)
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

    print("\n--- Phase 1: Loading and Validating Product Data ---")
    
    try:
        # Load root products_cleaned.json (has Handle, Title, Price, Image Src)
        print(f"Loading main product data from: {CLEANED_PRODUCTS_JSON_PATH}")
        if not os.path.exists(CLEANED_PRODUCTS_JSON_PATH):
            print(f"❌ Error: {CLEANED_PRODUCTS_JSON_PATH} does not exist!")
            sys.exit(1)
            
        root_products = load_products(CLEANED_PRODUCTS_JSON_PATH)
        print(f"Loaded {len(root_products)} products from root products_cleaned.json")
        
        # Load data/products_cleaned.json (has product_id, vendor, category)
        print(f"Loading vendor/category data from: {DATA_PRODUCTS_CLEANED_PATH}")
        if not os.path.exists(DATA_PRODUCTS_CLEANED_PATH):
            print(f"❌ Error: {DATA_PRODUCTS_CLEANED_PATH} does not exist!")
            sys.exit(1)
            
        with open(DATA_PRODUCTS_CLEANED_PATH, 'r', encoding='utf-8') as f:
            data_products = json.load(f)
        print(f"Loaded {len(data_products)} products from data/products_cleaned.json")
        
        # Check for matching handles
        root_handles = set(p.get('Handle') for p in root_products if p.get('Handle'))
        data_handles = set(p.get('product_id') for p in data_products if p.get('product_id'))
        matching_handles = root_handles.intersection(data_handles)
        
        print(f"Root products with Handle: {len(root_handles)}")
        print(f"Data products with product_id: {len(data_handles)}")
        print(f"Matching handles: {len(matching_handles)}")
        
        if len(matching_handles) == 0:
            print("❌ ERROR: No matching handles between files!")
            print(f"Sample root handles: {list(root_handles)[:5]}")
            print(f"Sample data handles: {list(data_handles)[:5]}")
            sys.exit(1)
        
        # Create mapping from product_id to vendor/category/image_file
        vendor_category_map = {}
        for product in data_products:
            product_id = product.get('product_id')
            if product_id and product_id in root_handles:  # Only include matching ones
                vendor_category_map[product_id] = {
                    'vendor': product.get('vendor'),
                    'category': product.get('category'),
                    'image_file': product.get('image_file')
                }
        
        print(f"Created vendor/category mapping for {len(vendor_category_map)} matching products")
        
        # Filter root products to only those that have matching data
        products_to_process = [p for p in root_products if p.get('Handle') in vendor_category_map]
        print(f"Products to process: {len(products_to_process)}")
        
        if len(products_to_process) == 0:
            print("❌ ERROR: No products to process after filtering!")
            sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"❌ Error: Product JSON file was not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
        sys.exit(1)

    print("\n--- Phase 2: Downloading Product Images ---")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for product in tqdm(products_to_process, desc="Downloading images"):
        handle = product.get('Handle')
        image_url = product.get('Image Src')

        if handle and image_url:
            vendor_data = vendor_category_map.get(handle)
            if vendor_data and vendor_data.get('image_file'):
                image_file = vendor_data['image_file']
            else:
                image_file = f"{handle}.jpg"
            
            image_path = os.path.join(IMAGE_DIR, image_file)
            
            if not os.path.exists(image_path):
                success = download_image(image_url, image_path)
                if success:
                    downloaded += 1
                else:
                    failed += 1
                    print(f"\nWarning: Failed to download image for Handle: {handle}")
            else:
                skipped += 1
        else:
            failed += 1
            print(f"\nWarning: Missing Handle or Image Src for product. Skipping.")
    
    print(f"\nImage download summary:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Failed: {failed}")

    print("\n--- Phase 3: Extracting Features and Building Vector Index ---")
    try:
        extractor = EfficientNetFeatureExtractor()
        actual_feature_dim = extractor.feature_dim 
        
        with open(DIMENSION_PATH, 'w') as f:
            f.write(str(actual_feature_dim))

        vector_db = VectorDB()

        features = []
        handles = []
        
        processed = 0
        skipped_missing_image = 0
        skipped_no_features = 0
        
        for product in tqdm(products_to_process, desc="Extracting features"):
            handle = product.get('Handle')
            
            if not handle:
                continue
            
            vendor_data = vendor_category_map.get(handle)
            if vendor_data and vendor_data.get('image_file'):
                image_file = vendor_data['image_file']
            else:
                image_file = f"{handle}.jpg"
                
            image_path = os.path.join(IMAGE_DIR, image_file)
            
            if not os.path.exists(image_path):
                skipped_missing_image += 1
                continue
                
            try:
                feature = extractor.get_features(image_path)
                features.append(feature)
                handles.append(handle)
                processed += 1
            except Exception as e:
                skipped_no_features += 1
                print(f"\nWarning: Could not extract features for {handle}: {e}")
                continue 

        print(f"\nFeature extraction summary:")
        print(f"  Successfully processed: {processed}")
        print(f"  Skipped (missing image): {skipped_missing_image}")
        print(f"  Skipped (feature extraction failed): {skipped_no_features}")

        if not features:
            print("❌ No features extracted. Index cannot be built.")
            sys.exit(1)

        print(f"Building index with {len(features)} products...")
        vector_db.build_index(features, handles, actual_feature_dim) 
        print("✅ Indexing complete!")

    except Exception as e:
        print(f"❌ An error occurred during feature extraction or indexing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print(f"\n🎉 Setup Complete! Successfully indexed {len(features)} products.")
    print("You can now start the Flask app with: python app.py")

if __name__ == '__main__':
    run_setup()