# utils/data_loader.py
import json
import os
import requests
from config import IMAGE_DIR, PRODUCTS_JSON_PATH, CLEANED_PRODUCTS_JSON_PATH

def load_products(json_path=PRODUCTS_JSON_PATH):
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

def load_cleaned_products(json_path=CLEANED_PRODUCTS_JSON_PATH):
    """
    Loads cleaned product data from a JSON file.
    Args:
        json_path (str): The path to the cleaned JSON file.
    Returns:
        list: A list of cleaned product dictionaries.
    """
    return load_products(json_path)
