# app.py (Modified Functions)
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import json 
from flask_cors import CORS 

# Local imports from our modular structure
from models.efficientnet_extractor import EfficientNetFeatureExtractor
from models.vector_db import VectorDB
from utils.data_loader import load_and_merge_products # Import the new function
from config import (
    UPLOAD_DIR, BASE_DIR, TOP_K_DISPLAY, TOP_K_CANDIDATES,
    ANNOY_INDEX_PATH, HANDLE_PATH, DIMENSION_PATH, IMAGE_DIR,
    PRODUCTS_JSON_PATH, CLEANED_PRODUCTS_JSON_PATH
)

app = Flask(__name__)
CORS(app) 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

feature_extractor = None
vector_db = None
all_products_metadata = {}
unique_product_types = []
unique_vendors = []

def init_app_resources():
    global feature_extractor, vector_db, all_products_metadata, unique_product_types, unique_vendors

    print("Initializing application resources...")
    
    feature_extractor = EfficientNetFeatureExtractor()
    print(f"Feature Extractor initialized. Expected feature dimension: {feature_extractor.feature_dim}")

    vector_db = VectorDB()
    
    if os.path.exists(ANNOY_INDEX_PATH) and \
       os.path.exists(HANDLE_PATH) and \
       os.path.exists(DIMENSION_PATH):
        try:
            vector_db.load_index()
            if vector_db.loaded_dimension != feature_extractor.feature_dim:
                print(f"Warning: Loaded Annoy index dimension ({vector_db.loaded_dimension}) does not match feature extractor dimension ({feature_extractor.feature_dim}). Please re-run `python setup.py` to rebuild the index after running `clean_product_data.py`.")
            print("VectorDB index loaded.")
        except FileNotFoundError as e:
            print(f"Error loading index: {e}. Please run `python setup.py` first to build the index.")
            vector_db = None 
        except Exception as e:
            print(f"Error loading Annoy index (possibly corrupted/mismatched): {e}")
            print("Please delete `data/product_index.ann`, `data/handles.npy`, `data/feature_dimension.txt` and re-run `python setup.py`.")
            vector_db = None
    else:
        print("Annoy index files (product_index.ann, handles.npy, feature_dimension.txt) not found. Please run `python setup.py` to build the index before starting the server.")
        vector_db = None 

    try:
        # Using the new merge function
        full_products_data = load_and_merge_products(PRODUCTS_JSON_PATH, CLEANED_PRODUCTS_JSON_PATH) 
        
        all_products_metadata = {p.get('Handle'): p for p in full_products_data if p.get('Handle')}
        print(f"Loaded {len(all_products_metadata)} product metadata entries from merged JSON.")

        types_set = set()
        vendors_set = set()
        for product in all_products_metadata.values():
            product_type = product.get('Type')
            vendor = product.get('Vendor')
            if product_type and product_type.strip() and product_type.strip().lower() != 'other':
                types_set.add(product_type.strip())
            if vendor and vendor.strip() and vendor.strip().lower() != 'unknown':
                vendors_set.add(vendor.strip())
        
        unique_product_types = sorted(list(types_set))
        unique_vendors = sorted(list(vendors_set))
        print(f"Discovered unique Types for filtering: {unique_product_types}")
        print(f"Discovered unique Vendors for filtering: {unique_vendors}")
    except FileNotFoundError:
        print("Error: One of the product JSON files was not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in one of the product files.")
    except Exception as e:
        print(f"An unexpected error occurred during product metadata loading: {str(e)}")
    
    print("Application resources initialization complete.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_filter_options', methods=['GET'])
def get_filter_options():
    return jsonify({
        "product_types": unique_product_types,
        "vendors": unique_vendors
    })

@app.route('/search', methods=['POST'])
def search():
    if not vector_db or not feature_extractor:
        return jsonify({"error": "Service not ready. Index or feature extractor not loaded. Please run `python setup.py` after `clean_product_data.py`."}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image provided in the request."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename provided."}), 400
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        file.save(temp_path)
        print(f"Received and saved user image to {temp_path}")
        
        query_features = feature_extractor.get_features(temp_path)
        
        initial_match_handles = vector_db.search(query_features, k=TOP_K_CANDIDATES) 

        initial_matches_full_info = []
        for handle in initial_match_handles:
            product_info = all_products_metadata.get(handle)
            if product_info:
                initial_matches_full_info.append(product_info)

        filtered_matches = []
        requested_type = request.form.get('product_type') 
        requested_vendor = request.form.get('vendor')     
        
        for product in initial_matches_full_info:
            passes_type_filter = True
            passes_vendor_filter = True

            if requested_type and requested_type != "All": 
                if product.get('Type', '').strip().lower() != requested_type.lower():
                    passes_type_filter = False
            
            if requested_vendor and requested_vendor != "All":
                if product.get('Vendor', '').strip().lower() != requested_vendor.lower():
                    passes_vendor_filter = False
            
            if passes_type_filter and passes_vendor_filter:
                filtered_matches.append(product)
        
        final_matches_to_return = filtered_matches[:TOP_K_DISPLAY]

        response_matches = []
        for product_info in final_matches_to_return:
            price_display = f"${product_info.get('Price', 'N/A')}" if product_info.get('Price') else '$N/A'
            response_matches.append({
                "Handle": product_info.get("Handle"),
                "Title": product_info.get("Title", "Untitled Product"),
                "Vendor": product_info.get("Vendor", "N/A"),
                "Price": price_display,
                "Image_Src": product_info.get("Image_Src"),
                "Type": product_info.get("Type", "N/A")
            })

        os.remove(temp_path)
        
        return jsonify({
            "status": "success",
            "matches": response_matches,
            "count": len(response_matches),
        }), 200

    except FileNotFoundError as fne:
        return jsonify({"error": f"File error: {str(fne)}"}), 500
    except ValueError as ve:
        return jsonify({"error": f"Image processing error: {str(ve)}"}), 400
    except Exception as e:
        print(f"An unexpected error occurred during search: {str(e)}") 
        return jsonify({"error": f"An unexpected error occurred during search: {str(e)}."}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/data/product_images/<path:filename>')
def serve_product_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    init_app_resources()
    app.run(host='0.0.0.0', port=5000, debug=False)