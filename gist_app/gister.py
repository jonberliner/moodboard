import os
import pandas as pd
from typing import Optional, List

import numpy as np
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


from utils.all_utils import get_data_dir
from product import Product
from product_set import ProductSet, ProductSetFactory

class Gister:

    def __init__(self) -> None:

        # Store our products here
        self._product_set: Optional[ProductSet] = None

        # We'll hold the model for embedding creation here
        #   tweaked from https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel.forward.returns
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @property
    def product_set(self) -> Optional[ProductSet]:
        return self._product_set
    
    @product_set.setter
    def product_set(self, products: ProductSet) -> None:
        self._products = products

    # Helper to get the number of products we have
    def get_num_products(self) -> int:
        if self._product_set is None:
            return 0
        return self._product_set.get_num_products()

    # Helper to get the number of embeddings we have
    def get_num_product_embeddings(self) -> int:
        if self._product_set is None:
            return 0
        return self._product_set.get_num_product_embeddings()

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


    def load_product_set(self, p_type: str, data_source: str = 'local', preload_all: bool = False, use_prebuilt: bool = False) -> None:

        # If we're using a saved product set, load it and return
        if use_prebuilt:
            self._product_set = ProductSetFactory.load_product_set(p_type, data_source)
            if self._product_set is not None:
                return

            # Warn if we didn't find it
            print(f"WARNING: Could not find prebuilt product set for {p_type} in {data_source}")

        # First, create the product set
        self._product_set = ProductSetFactory.create_product_set(p_type, data_source)

        # Then load the products
        self._product_set.load_products()

        # Then the images
        self._product_set.load_images(preload_all)

        # And finally the embeddings
        self._product_set.load_embeddings(self)


    # Search for a product
    def search_image(self, image: Image, category: str, num_results: int = 10) -> List[Product]:

        # Error if we have no products
        if self.product_set is None:
            raise ValueError("No product set loaded")

        # And search the product set
        return self.product_set.search_image(image, category, num_results, self)

    
    # Search for a product by image url
    def search_image_url(self, image_url: str, category: str, num_results: int = 10) -> List[Product]:
            
        # Get the image
        image = Image.open(requests.get(image_url, stream=True).raw)
    
        # Search
        return self.search_image(image, category, num_results)
    
    # Helper to return the product categories
    def get_product_categories(self) -> List[str]:

        # Return 0 if no products
        if self.product_set is None:
            return []
        
        # Otherwise, return the categories
        return self.product_set.get_categories()
