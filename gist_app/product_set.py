# A base class for all of our different product sets

import os
import numpy as np
import torch
from typing import List
import faiss
from PIL import Image
import io

from utils.all_utils import get_data_dir, get_s3_resource, get_s3_bucket_name
from product import Product

class ProductSet:

    def __init__(self, data_source: str) -> None:
        self.data_source = data_source
        pass

    # Create a virtual load images method
    def load_images(self, preload_all=False) -> None:
        raise NotImplementedError

    # Create a virtual load products method
    def load_products(self) -> None:
        raise NotImplementedError

    # Create a virtual function to get the number of products
    def get_num_products(self) -> int:
        raise NotImplementedError

    # Virtual function to get the categories
    def get_categories(self) -> List[str]:
        raise NotImplementedError

    # Create a virtual function to get the embeddings path
    def get_embeddings_path(self) -> str:
        raise NotImplementedError

    # Create a virtual function to create embeddings
    def create_embeddings(self, gister) -> List[torch.Tensor]:
        raise NotImplementedError

    # Virtual function to get a product
    def get_product(self, idx: int) -> Product:
        raise NotImplementedError

    # Create a virtual load embeddings method
    def load_embeddings(self, gister) -> None:
        
        # Error if we have no products
        if self.get_num_products() == 0:
            raise ValueError("No products loaded")

        # Load the embeddings, if we have them saved
        embeddings_path = self.get_embeddings_path()
        if self.data_source == 'local':

            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                return

        # Check s3
        elif self.data_source == 's3':

            # Load if we can
            s3 = get_s3_resource()
            print("Loading embeddings from S3...")
            try:

                # Get the embeddings object
                obj = s3.Object(get_s3_bucket_name(), embeddings_path).get()

                # Convert to numpy
                self.embeddings = np.load(io.BytesIO(obj['Body'].read()))
                return

            except:
                # Error if we can't
                raise ValueError("Embeddings not loaded in S3")

        # Otherwise, error
        else:
            raise ValueError("Invalid data source")


        # If not, create them
        candidate_vembeds = self.create_embeddings(gister)

        # Format and save the embeddings
        candidate_vembeds = torch.concat(candidate_vembeds)
        candidate_vembeds = candidate_vembeds.detach().numpy()
        np.save(embeddings_path, candidate_vembeds)

        # And load
        self.load_embeddings(gister)

    # Virtual fuction to get the category indices
    def get_category_indices(self, category: str) -> np.array:
        raise NotImplementedError

    # Return the number of product embeddings
    def get_num_product_embeddings(self) -> int:
        return len(self.embeddings)


    # Search for an image
    def search_image(self, image: Image, category: str, num_results: int, gister) -> List[Product]:

        # Get the embeddings
        with torch.no_grad():
            query_vembeds = gister.images_to_embeddings([image])
        query_vembeds = query_vembeds.detach().numpy()

        # Get the category embeddings
        category_idxs = self.get_category_indices(category)
        category_vembeds = self.embeddings[category_idxs]

        # Create the index
        dim = query_vembeds.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(category_vembeds)

        # Search the index
        _, idxs = index.search(query_vembeds, num_results)

        # Get the products
        products = []
        for idx in idxs[0]:
            products.append(self.get_product(category_idxs[idx]))
        return products


# ProductSet factory class
class ProductSetFactory:

    # Create a product set based on the name
    @staticmethod
    def create_product_set(p_type: str, data_source: str) -> ProductSet:

        # If amazon, load the amazon products
        if p_type == "amazon":
            
            from amazon_clothing_product_set import AmazonClothingProductSet

            # Download the data if we don't have it
            download_dir = AmazonClothingProductSet.get_download_path()
            fpath = os.path.join(download_dir, "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson")

            # Check the records file exists
            if not os.path.exists(fpath):
                AmazonClothingProductSet.download_records(download_dir, fpath)

            # Read the records
            records = AmazonClothingProductSet.read_records(fpath)

            # Create the dataset
            # Note: this will download the images
            return AmazonClothingProductSet(image_dir=download_dir, records=records)

        # Otherwise, could be asos
        elif p_type == "asos":
            from asos_product_set import AsosProductSet

            # Create the dataset
            return AsosProductSet(data_source=data_source)

        # Otherwise, error
        else:
            raise ValueError(f"Unknown product type: {p_type}")

