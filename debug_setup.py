# debug_setup.py - Run this to see what's happening
import os
import json
from config import CLEANED_PRODUCTS_JSON_PATH, DATA_PRODUCTS_CLEANED_PATH, IMAGE_DIR, ANNOY_INDEX_PATH, HANDLE_PATH
import numpy as np

def debug_setup():
    print("=== DEBUGGING SETUP PROCESS ===")
    
    # 1. Check files exist
    print(f"\n1. FILE EXISTENCE CHECK:")
    print(f"Root products_cleaned.json: {os.path.exists(CLEANED_PRODUCTS_JSON_PATH)} - {CLEANED_PRODUCTS_JSON_PATH}")
    print(f"Data products_cleaned.json: {os.path.exists(DATA_PRODUCTS_CLEANED_PATH)} - {DATA_PRODUCTS_CLEANED_PATH}")
    print(f"Image directory: {os.path.exists(IMAGE_DIR)} - {IMAGE_DIR}")
    
    # 2. Check data loading
    print(f"\n2. DATA LOADING CHECK:")
    try:
        with open(CLEANED_PRODUCTS_JSON_PATH, 'r', encoding='utf-8') as f:
            root_products = json.load(f)
        print(f"✅ Root products loaded: {len(root_products)}")
        if root_products:
            print(f"   Sample Handle: {root_products[0].get('Handle')}")
            print(f"   Sample Image Src: {root_products[0].get('Image Src')}")
    except Exception as e:
        print(f"❌ Error loading root products: {e}")
        return
    
    try:
        with open(DATA_PRODUCTS_CLEANED_PATH, 'r', encoding='utf-8') as f:
            data_products = json.load(f)
        print(f"✅ Data products loaded: {len(data_products)}")
        if data_products:
            print(f"   Sample product_id: {data_products[0].get('product_id')}")
            print(f"   Sample image_file: {data_products[0].get('image_file')}")
    except Exception as e:
        print(f"❌ Error loading data products: {e}")
        return
    
    # 3. Check Handle matching
    print(f"\n3. HANDLE MATCHING CHECK:")
    root_handles = set(p.get('Handle') for p in root_products if p.get('Handle'))
    data_handles = set(p.get('product_id') for p in data_products if p.get('product_id'))
    
    print(f"Root handles count: {len(root_handles)}")
    print(f"Data handles count: {len(data_handles)}")
    
    matching_handles = root_handles.intersection(data_handles)
    print(f"Matching handles: {len(matching_handles)}")
    
    if len(matching_handles) == 0:
        print("❌ NO MATCHING HANDLES! This is the problem!")
        print(f"Sample root handles: {list(root_handles)[:5]}")
        print(f"Sample data handles: {list(data_handles)[:5]}")
        return
    else:
        print(f"✅ Found {len(matching_handles)} matching handles")
        print(f"Sample matching: {list(matching_handles)[:3]}")
    
    # 4. Check image files
    print(f"\n4. IMAGE FILES CHECK:")
    image_files = os.listdir(IMAGE_DIR) if os.path.exists(IMAGE_DIR) else []
    print(f"Images in directory: {len(image_files)}")
    if image_files:
        print(f"Sample images: {image_files[:3]}")
    
    # 5. Check index files
    print(f"\n5. INDEX FILES CHECK:")
    print(f"Annoy index exists: {os.path.exists(ANNOY_INDEX_PATH)}")
    print(f"Handles file exists: {os.path.exists(HANDLE_PATH)}")
    
    if os.path.exists(HANDLE_PATH):
        try:
            handles = np.load(HANDLE_PATH)
            print(f"Handles in index: {len(handles)}")
            print(f"Sample indexed handles: {handles[:3]}")
        except Exception as e:
            print(f"Error reading handles: {e}")

if __name__ == "__main__":
    debug_setup()