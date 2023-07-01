# Class to load in the asos clothing database

import json
import os
import requests
from PIL import Image
from io import BytesIO
from typing import List
import torch
import numpy as np

from utils.all_utils import get_data_dir
from product_set import ProductSet
from product import Product

class AsosProductSet(ProductSet):

    def __init__(self) -> None:

        # Set the base path
        self.data_path = AsosProductSet.get_data_path()

        # For now, only one category
        self.category = 'WomensDresses'

        # Will store everything here
        self.all_products = None
        self.images = None
        self.embeddings = None

    # Have a static function to return the base data path
    @staticmethod
    def get_data_path() -> str:
        return os.path.join(get_data_dir(),'asos')

    # Return our category indices
    def get_category_indices(self, category: str) -> np.array:

        # Note that we only have one category, so return all indices as a np.array
        return np.arange(self.get_num_products())

    # Return the embeddings path
    def get_embeddings_path(self) -> str:
        return os.path.join(self.data_path,'asos_embeddings.npy')

    # Load the products
    def load_products(self) -> None:

        # Error if we don't have the products
        products_path = os.path.join(self.data_path,'all_products.json')
        if not os.path.exists(products_path):
            raise ValueError("No products found")

        # Load the products
        with open(products_path, 'r') as f:
            self.all_products = json.load(f)
        print(f"Loaded {self.get_num_products()} products")

    # Return the number of products
    def get_num_products(self) -> int:
        return len(self.all_products)

    # Return the categories
    def get_categories(self) -> List[str]:
        return [self.category]

    # Create a virtual function to create embeddings
    def create_embeddings(self, gister) -> List[torch.Tensor]:

        # Create the embeddings
        candidate_vembeds = []

        print("Creating embeddings...")
        idx = 0
        batch_size = 100

        # Loop through the products creating batches
        while True:

            # Get the next batch
            batch = self.all_products[idx:idx+batch_size]

            # Might be done
            if len(batch) == 0:
                break
                
            # Create the batch of images
            batch_images = []
            for product in batch:
                img = Image.open(product['image_path'])
                batch_images.append(img)

            # And convert to a tensor
            with torch.no_grad():
                candidate_vembeds.append(gister.images_to_embeddings(batch_images))

            idx += batch_size
            print(f"Created {idx} embeddings")

        # And return
        return candidate_vembeds


    # Get a product
    def get_product(self, product_id: int) -> Product:

        # Make sure we have enough
        if product_id >= self.get_num_products():
            raise ValueError("Product ID out of range")

        # Get the product info
        product = self.all_products[product_id]

        # Load the image
        image_path = product['image_path']
        try:
            image = Image.open(image_path)

        # These should all be here
        except:
            raise ValueError(f"Could not load image {image_path}")
        
        # Return the product
        return Product(label=product['name'], image=image, category=self.category)


    # Load the images
    def load_images(self) -> None:

        # Get the image path
        images_dir = os.path.join(self.data_path,'images')

        # Create the images dir if it doesn't exist
        os.makedirs(images_dir, exist_ok=True)

        # Will be removing any products that don't have images
        new_products = []

        # Loop through the products and link or download the images
        print("Loading images...")
        for idx, product in enumerate(self.all_products):

            # Get the image url    
            image_url = product['imageUrl']

            # Get the local path
            ipath = os.path.join(images_dir, f'image_{idx}.jpg')

            # Download the image if we don't have it
            if not os.path.exists(ipath):
                try:
                    response = requests.get("https://" + image_url)
                    img = Image.open(BytesIO(response.content))
                    img.save(ipath)
                    print(f"Saved image {idx}")

                except:
                    print(f"WARNING: Failed to save image {idx}")

            # Link the image if we have it
            if os.path.exists(ipath):
                product['image_path'] = ipath
                new_products.append(product)

        # Update the products
        self.all_products = new_products
        print("Done loading images")


    #################################
    #
    # Static function to download the data
    #
    ##################################

    # A static function for getting the category list
    @staticmethod
    def download_category_list() -> List[str]:

        # Get the download path
        cpath = os.path.join(AsosProductSet.get_data_path(),'categories.json')

        # Load the categories if we have them
        if os.path.exists(cpath):
            with open(cpath, 'r') as f:
                return json.load(f)

        # Get the RapidAPI key from the environment
        rapidapi_key = os.getenv('RAPIDAPI_KEY')

        # Otherwise, download them
        url = "https://asos2.p.rapidapi.com/categories/list"
        querystring = {"country":"US","lang":"en-US"}
        headers = {
            "X-RapidAPI-Key": rapidapi_key,
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }

        # NOTE: Can dive deep into link: link-type as category in children
        # ...or can go to asos and click around, and the category id is in the url
        # E.g. https://www.asos.com/us/women/dresses/cat/?cid=8799 is dresses

        response = requests.get(url, headers=headers, params=querystring)

        # Save the response as a json file
        with open(cpath, 'w') as f:
            f.write(response.text)

        # And load and return
        with open(cpath, 'r') as f:
            return json.load(f)


    # Static function to get the product details    
    @staticmethod
    def download_product_details(product_id: str) -> List[str]:

        # Get the download path
        cpath = os.path.join(AsosProductSet.get_data_path(),f'product_{product_id}.json')

        # Load and return if we already have it
        if os.path.exists(cpath):
            with open(cpath, 'r') as f:
                return json.load(f)

        url = "https://asos2.p.rapidapi.com/products/v3/detail"

        querystring = {"id":product_id,"lang":"en-US","store":"US","sizeSchema":"US","currency":"USD"}

        headers = {
            "X-RapidAPI-Key": os.getenv('RAPIDAPI_KEY'),
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)

        # Save the response as a json file
        with open(cpath, 'w') as f:
            f.write(response.text)

        # And load and return
        with open(cpath, 'r') as f:
            return json.load(f)

    # A static function to get the products
    @staticmethod
    def download_products(category_id: str) -> List[str]:

        # Get the download path
        cpath = os.path.join(AsosProductSet.get_data_path(),f'all_product_{category_id}.json')

        # Load and return if we already have it
        if os.path.exists(cpath):
            with open(cpath, 'r') as f:
                return json.load(f)

        # Otherwise, download in batches
        url = "https://asos2.p.rapidapi.com/products/v2/list"

        headers = {
            "X-RapidAPI-Key": os.getenv('RAPIDAPI_KEY'),
            "X-RapidAPI-Host": "asos2.p.rapidapi.com"
        }

        # Accumulate the products in a list
        all_products = []

        # Run through and increase the offset by 48 each time
        offset = 0
        while True:
            
            # Get the save path
            opath = os.path.join(AsosProductSet.get_data_path(),f'all_product_{category_id}_{offset}.json')
            
            # Download if we don't have it
            if not os.path.exists(opath):

                querystring = {"store":"US","offset":offset,"categoryId":category_id,"limit":"48","country":"US","sort":"freshness","currency":"USD","sizeSchema":"US","lang":"en-US"}
                response = requests.get(url, headers=headers, params=querystring)
        
                # Make sure we have some products left
                num_products = len(response.json()['products'])
                if num_products == 0:
                    print(f"No more products at offset {offset}")
                    break

                # Save the response as a json file, so we don't have to redo
                with open(opath, 'w') as f:

                    # Save the response
                    f.write(response.text)

                # Log
                print(f"Saved products at offset {offset}")

            # Load
            with open(opath, 'r') as f:
                response_json = json.load(f)

            # Add to the list
            all_products += response_json['products']

            # And iterate
            offset += 48

        # And save all the products in the asos folder
        with open(cpath, 'w') as f:

            # Save the all products list
            f.write(json.dumps(all_products))

        # And load the all products list
        with open(cpath, 'r') as f:
            return json.load(f)

