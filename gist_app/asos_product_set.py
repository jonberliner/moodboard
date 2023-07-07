# Class to load in the asos clothing database

import json
import os
import requests
from PIL import Image
from io import BytesIO
from typing import List
import torch
import numpy as np
import io

from utils.all_utils import get_data_dir, get_s3_resource, get_s3_bucket_name
from product_set import ProductSet
from product import Product

class AsosProductSet(ProductSet):

    def __init__(self, data_source: str) -> None:

        # Call the base class constructor
        super().__init__(data_source)

        # For now, only one category
        self.category = 'WomensDresses'
        self.category_id = '8799'

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

        # Get the base name
        e_name = f'product_embeddings_{self.category_id}.npy'

        # Different sources
        if self.data_source == 'local':
            return os.path.join(AsosProductSet.get_data_path(),e_name)

        elif self.data_source == 's3':
            return f'asos/{e_name}'
        else:
            raise ValueError(f"Unknown source {self.data_source}")

    # Load the products
    def load_products(self) -> None:

        # We're named after the product id
        file_name = f'all_products_{self.category_id}.json'

        if self.data_source == 'local':

            # Error if we don't have the products
            products_path = os.path.join(AsosProductSet.get_data_path(),file_name)
            if not os.path.exists(products_path):
                raise ValueError("No products found")

            # Load the products
            with open(products_path, 'r') as f:
                self.all_products = json.load(f)

        elif self.data_source == 's3':

            # get the s3 resource
            s3 = get_s3_resource()

            # Get an object from the bucket
            print("Loading products from s3...")
            obj = s3.Object(get_s3_bucket_name(), 'asos/'+file_name)

            # And convert the contents to json
            self.all_products = json.loads(obj.get()['Body'].read().decode('utf-8'))

        else:
            raise ValueError(f"Unknown source {self.data_source}")

        # And log
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

        # Load if we don't already have it
        if 'image' not in product:
            self.load_image(product_id)

        # Return the product
        return Product(label=product['name'], image=product['image'], category=self.category, url='https://www.asos.com/'+product['url'])

    # Helper function to load an image
    def load_image(self, product_id):

        # Get the product and path
        product = self.all_products[product_id]
        image_path = product['image_path']

        try:

            # If local, just load
            if self.data_source == 'local':

                # Workaround for PIL bug
                temp = Image.open(image_path)
                image = temp.copy()
                temp.close()

            # If s3, then retrieve
            elif self.data_source == 's3':
                s3 = get_s3_resource()
                obj = s3.Object('gist-data', image_path).get()
                image = Image.open(io.BytesIO(obj['Body'].read()))

            # Otherwise, error
            else:
                raise ValueError(f"Unknown source {self.data_source}")

        # These should all be here
        except:
            raise ValueError(f"Could not load image {image_path}")
        
        # Store so we don't have to reload later
        self.all_products[product_id]['image'] = image        


    # Load the images
    def load_images(self, preload_all=False) -> None:

        # If we're local, then create the folder
        if self.data_source == 'local':

            # Get the image path
            images_dir = os.path.join(AsosProductSet.get_data_path(),'images',self.category_id)

            # Create the images dir if it doesn't exist
            os.makedirs(images_dir, exist_ok=True)

        # Otherwise, we'll be downloading from s3
        elif self.data_source == 's3':

            # The path to the images
            images_dir = 'asos/images/'+self.category_id+'/'

            # And get the resource
            s3 = get_s3_resource()

            # Get the bucket
            bucket = s3.Bucket(get_s3_bucket_name())

            # For speed, get all the files in the directory now
            s3_files = [obj.key for obj in bucket.objects.filter(Prefix=images_dir)]

            # And remove the directory name
            s3_files = [file.split('/')[-1] for file in s3_files]

        # Otherwise, error
        else:
            raise ValueError(f"Unknown source {self.data_source}")

        # Will be removing any products that don't have images
        new_products = []

        # Loop through the products and link or download the images
        print("Loading images...")
        for idx, product in enumerate(self.all_products):

            # Looking for this image name
            image_name = f'image_{idx}.jpg'

            # Get the image url    
            image_url = product['imageUrl']

            # If we're local, try to find or download the image
            if self.data_source == 'local':

                # Get the local path
                ipath = os.path.join(images_dir, image_name)

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


            # Otherwise, check s3 to see if we're there
            elif self.data_source == 's3':

                # Check the s3 files
                if image_name in s3_files:

                    # Record that we have it
                    product['image_path'] = images_dir+image_name
                    new_products.append(product)

                # If this fails, no option to download for now
                else:
                    print(f"WARNING: Failed to find image {idx}")
            
            # Otherwise error
            else:
                raise ValueError(f"Unknown source {self.data_source}")

            # See if we're preloading all
            if preload_all and 'image_path' in product:
                print(f"Loading image {idx}")
                self.load_image(idx)

        # Update the products
        self.all_products = new_products
        print(f"Loaded {self.get_num_products()} images") 


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

