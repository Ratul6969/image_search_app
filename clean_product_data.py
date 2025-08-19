import json
import os
import requests
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import GEMINI_API_KEY, DATA_DIR

# Base directory for product images
product_images_dir = os.path.join(DATA_DIR, "product_images")

# The URL for the Gemini API
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
api_url = f"{API_BASE_URL}/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

def call_gemini_api(prompt, image_path=None, max_retries=5, initial_delay=1):
    """
    Calls the Gemini API with a given prompt and an optional image.
    Implements exponential backoff for retries to handle connection errors.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": []}

    # Add text prompt
    if prompt:
        payload["contents"].append({"role": "user", "parts": [{"text": prompt}]})

    # Add image if provided
    if image_path:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode("utf-8")
                payload["contents"].append({"role": "user", "parts": [{"inlineData": {"mimeType": "image/jpeg", "data": image_base64}}]})
        else:
            print(f"Warning: Image file not found at {image_path}. Skipping image part.")

    # Implement exponential backoff for robust API calls
    retries = 0
    delay = initial_delay
    response = None

    while retries < max_retries:
        try:
            # Make the API request with a timeout
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status() # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.ConnectionError as e:
            retries += 1
            print(f"Connection error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2 # Exponential backoff
        except requests.exceptions.HTTPError as e:
            # Handle specific HTTP errors if needed
            print(f"HTTP Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    print(f"Failed to call Gemini API after {max_retries} retries.")
    return None

def infer_vendor_from_title(title):
    """Uses Gemini API to infer a vendor/brand from a product title."""
    prompt = f"Extract the brand or manufacturer's name from this product title, return just the name: '{title}'"
    response_json = call_gemini_api(prompt)
    
    if response_json and "candidates" in response_json and len(response_json["candidates"]) > 0:
        inferred_vendor = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Simple cleanup
        if inferred_vendor.lower() in ["unbranded", "generic", "no brand", "none"]:
            return "Unknown"
        return inferred_vendor
    return "Unknown"

def infer_category_from_title(title):
    """Uses Gemini API to infer a category from a product title."""
    prompt = f"Categorize the following product title into one of these categories: [Electronics, Home Goods, Apparel, Outdoors, Toys, Tools, Health & Beauty, Musical Instruments, Luggage]. Return only the category name. Title: '{title}'"
    response_json = call_gemini_api(prompt)

    if response_json and "candidates" in response_json and len(response_json["candidates"]) > 0:
        inferred_category = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Simple validation against known categories
        valid_categories = ["Electronics", "Home Goods", "Apparel", "Outdoors", "Toys", "Tools", "Health & Beauty", "Musical Instruments", "Luggage"]
        if inferred_category in valid_categories:
            return inferred_category
        return "Uncategorized"
    return "Uncategorized"

def clean_product_data():
    """
    Cleans and enhances product data by inferring vendor and category using the Gemini API.
    Uses a thread pool to process API calls concurrently.
    """
    try:
        # Load products.json from the root directory
        with open("products.json", "r", encoding="utf-8") as f:
            products = json.load(f)
    except FileNotFoundError:
        print("Error: products.json not found. Make sure the file exists in the root directory.")
        return

    cleaned_products = []
    
    # Use a thread pool for concurrent API calls to speed up the process
    # The number of workers can be adjusted based on your system and API rate limits
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_product, product): product for product in products}

        # Use tqdm to show a progress bar
        for future in tqdm(as_completed(futures), total=len(products), desc="Cleaning and enriching product data"):
            cleaned_product = future.result()
            if cleaned_product:
                cleaned_products.append(cleaned_product)
    
    # Write the cleaned data to the data directory as originally intended
    with open(os.path.join(DATA_DIR, "products_cleaned.json"), "w", encoding="utf-8") as f:
        json.dump(cleaned_products, f, indent=4)

    print("\nData cleaning and enrichment complete. Cleaned data saved to products_cleaned.json")

def process_product(product):
    """
    Processes a single product for cleaning and enrichment, preserving all original keys.
    """
    cleaned_product = product.copy()

    product_title = cleaned_product.get("Title", "")
    
    if not product_title:
        print(f"Skipping product with missing title: {cleaned_product}")
        return None
    
    # Add the inferred vendor and category to the existing dictionary
    cleaned_product["Vendor"] = infer_vendor_from_title(product_title)
    cleaned_product["Product Category"] = infer_category_from_title(product_title)
    
    return cleaned_product

if __name__ == "__main__":
    clean_product_data()
