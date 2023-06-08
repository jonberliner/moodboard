import os
import pandas as pd
from typing import Optional, List

import faiss
import numpy as np
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


from amazon_clothing_database import AmazonClothingDataset
from utils.all_utils import get_data_dir
from product import Product

class Gister:

    def __init__(self) -> None:

        # Store our products here
        self._products: Optional[AmazonClothingDataset] = None

        # We'll hold the model for embedding creation here
        #   tweaked from https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel.forward.returns
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @property
    def products(self) -> Optional[AmazonClothingDataset]:
        return self._products
    
    @products.setter
    def products(self, products: AmazonClothingDataset) -> None:
        self._products = products

    # Helper to get the number of products we have
    def get_num_products(self) -> int:
        if self._products is None:
            return 0
        return len(self._products)

    # Helper to get the number of embeddings we have
    def get_num_product_embeddings(self) -> int:
        if self._products is None:
            return 0
        if self._products.embeddings is None:
            return 0
        return len(self._products.embeddings)

    # Create image embeddings
    #   Below is standard CLIP usage to score text snippets against a photo
    def images_to_embeddings(self, images: list[Image]) -> torch.tensor:

        # Get the input and output to the model
        vinput = self.processor(images=images, return_tensors="pt")

        voutput = self.model.vision_model(
            **vinput
        )

        # Create the embedding from the output
        vembeds = voutput[1]
        vembeds = self.model.visual_projection(vembeds)
        vembeds = vembeds / vembeds.norm(p=2, dim=-1, keepdim=True)

        # And return        
        return vembeds

    # Helper to load the embeddings
    def load_product_embeddings(self, embeddings_path: str) -> None:
        
        # Error if we have no products
        if self._products is None:
            raise ValueError("No products loaded")

        # Check if we have the embeddings
        if not os.path.exists(embeddings_path):
            
            # If not, create them
            candidate_vembeds = []

            # Use the data loader
            count = 0
            for imgs, labs, cats in tqdm(self.products.get_data_loader()):
                count += len(imgs)
                with torch.no_grad():
                    candidate_vembeds.append(self.images_to_embeddings(imgs))

             # Concatenate the embeddings
            candidate_vembeds = torch.concat(candidate_vembeds)
            candidate_vembeds = candidate_vembeds.detach().numpy()

            # Save the embeddings
            np.save(embeddings_path, candidate_vembeds)

        # Load the embeddings
        self.products.embeddings = np.load(embeddings_path)


    def load_products(self) -> None:

        # Download the data if we don't have it
        download_dir = os.path.join(get_data_dir(),'amazon_clothing_2020')
        fpath = os.path.join(download_dir, "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson")

        # Check the records file exists
        if not os.path.exists(fpath):
            AmazonClothingDataset.download_records(download_dir, fpath)

        # Read the records
        records = AmazonClothingDataset.read_records(fpath)

        # Load (or download) the images
        self.products = AmazonClothingDataset(image_dir=download_dir, records=records)

        # Load the embeddings
        self.load_product_embeddings(os.path.join(download_dir, "amazon_embeddings.npy"))


    # Get a product
    def get_product(self, product_id: int) -> Product:

        # Error if we have no products
        if self.products is None:
            raise ValueError("No products loaded")

        # Get the product
        return Product(self.products[product_id][0], self.products[product_id][1], self.products[product_id][2])
    
    # Search for a product
    def search_image(self, image: Image, category: str, num_results: int = 10) -> List[Product]:

        # Error if we have no products
        if self.products is None:
            raise ValueError("No products loaded")

        # Get the embeddings
        with torch.no_grad():
            query_vembeds = self.images_to_embeddings([image])
        query_vembeds = query_vembeds.detach().numpy()

        # Get the category embeddings
        category_idxs = self.products.get_category_indices(category)
        category_vembeds = self.products.embeddings[category_idxs]

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
    
    # Search for a product by image url
    def search_image_url(self, image_url: str, category: str, num_results: int = 10) -> List[Product]:
            
        # Get the image
        image = Image.open(requests.get(image_url, stream=True).raw)
    
        # Search
        return self.search_image(image, category, num_results)