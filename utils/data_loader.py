# utils/data_loader.py
import json
import os

def load_products(json_path):
    """
    Loads product data from a JSON file.
    Args:
        json_path (str): The path to the JSON file.
    Returns:
        list: A list of product dictionaries.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
        return products
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file {json_path} contains invalid JSON.")
        return []

def load_and_merge_products(products_json_path, cleaned_products_json_path):
    """
    Loads data from two JSON files and merges them.
    Prioritizes specific fields from the cleaned data.
    """
    products_data = load_products(products_json_path)
    cleaned_data = load_products(cleaned_products_json_path)

    cleaned_map = {p.get('product_id'): p for p in cleaned_data}
    merged_products = {}

    for product in products_data:
        handle = product.get('Handle')
        if handle and handle in cleaned_map:
            cleaned_info = cleaned_map[handle]
            # Merge data, prioritizing cleaned data for vendor and category
            merged_product = {
                "Handle": handle,
                "Title": product.get("Title"),
                "Vendor": cleaned_info.get("vendor", product.get("Vendor")),
                "Type": cleaned_info.get("category", product.get("Type")),
                "Price": product.get("Variant Price"),
                "Image_Src": product.get("Image Src", "").strip()
            }
            merged_products[handle] = merged_product
        else:
            # Use original data if no cleaned data is found for the handle
            merged_products[handle] = {
                "Handle": handle,
                "Title": product.get("Title"),
                "Vendor": product.get("Vendor"),
                "Type": product.get("Type"),
                "Price": product.get("Variant Price"),
                "Image_Src": product.get("Image Src", "").strip()
            }
    
    return list(merged_products.values())