# utils/image_utils.py
import requests
from PIL import Image
from io import BytesIO
import os
from config import IMAGE_DIR, IMAGE_SIZE 

def download_product_images(products_data):
    """
    Downloads product images from the provided list of product dictionaries.
    Each product dict must have an 'Image Src' and 'Handle' key.
    Images are saved to the directory specified by config.IMAGE_DIR.
    
    Args:
        products_data (list of dict): A list of product dictionaries.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True) # Ensure directory exists
    
    print(f"Starting image download to {IMAGE_DIR}...")
    for product in products_data: # Iterate over the full raw products data
        handle = product.get('Handle')
        image_url = product.get('Image Src', '').strip() 

        if not handle or not image_url:
            print(f"Skipping product with missing Handle or Image Src: {product.get('Title', 'N/A')}")
            continue

        file_path = os.path.join(IMAGE_DIR, f"{handle}.jpg")

        if os.path.exists(file_path):
            # print(f"Skipping existing image for {handle}") # Comment out to reduce log noise
            continue 

        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE)) # Resize for consistent storage (optional)
            img_resized.save(file_path)
            
            print(f"✅ Saved image for {handle}")
        except requests.exceptions.RequestException as req_err:
            print(f"❌ Network error downloading {handle} from {image_url}: {req_err}")
        except Exception as e:
            print(f"❌ Failed to process/save image for {handle} from {image_url}: {str(e)}")