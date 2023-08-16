# A base class for all of our different product sets

from __future__ import annotations

import os
import numpy as np
import pickle
import torch
from typing import List
import faiss
from PIL import Image
import io
import random

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
    def get_embeddings_path(self, gister) -> str:
        raise NotImplementedError

    # Create a virtual function to create embeddings
    def create_embeddings(self, gister) -> List[torch.Tensor]:
        raise NotImplementedError

    # Virtual function to get a product
    def get_product(self, idx: int, should_load_image: bool = True) -> Product:
        raise NotImplementedError

    # Create a virtual load embeddings method
    def load_embeddings(self, gister) -> None:
        
        # Error if we have no products
        if self.get_num_products() == 0:
            raise ValueError("No products loaded")

        # Load the embeddings, if we have them saved
        embeddings_path = self.get_embeddings_path(gister)
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
    def search_images(self, images: List[Image], category: str, num_results: int, weight, 
                      search_text, text_weight, eval_adj, gister, should_load_images: bool = True) -> List[Product]:

        # Get the embeddings
        with torch.no_grad():
            query_vembeds = gister.images_to_embeddings(images)

        # If only one image, then take the vector directly
        if len(images) == 1:
            pass
          
        # Otherwise, take the weighted average
        elif len(images) == 2:

            # Need a weight if we have two images
            if weight is None:
                raise ValueError("Must provide a weight for two images")

            # Take the weighted average
            query_vembeds = (weight * query_vembeds[0]) + ((1. - weight) * query_vembeds[1]).unsqueeze(0)

        # Otherwise, error
        else:

            # TODO: Generalize this to more than 2 images
            raise ValueError("Invalid number of images")        

        # And add in the text if we have it
        if search_text is not None:
            with torch.no_grad():
                text_vembeds = gister.texts_to_embeddings([search_text])
            query_vembeds += text_vembeds[0] * text_weight

        # Get the category embeddings
        category_idxs = self.get_category_indices(category)
        category_vembeds = self.embeddings[category_idxs]
        # print(len(category_idxs))

        # we are going to remove items similar to things we said no to

        # Create the index
        dim = query_vembeds.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(category_vembeds)



        seen = set()
        to_remove = []
        to_keep = []
        # And any adjustment
        if eval_adj is not None:
            query_vembeds = torch.tensor(query_vembeds)
            to_add = None
            for yn, cat_idx, vect in eval_adj:
                vect = torch.tensor(vect)
                # if idx in seen:
                #     continue

                residual = vect - query_vembeds

                # seen.add(idx)
                if yn == 'Y':
                    query_vembeds += residual * 0.5
                    # query_vembeds = query_vembeds / query_vembeds.norm(p=2, dim=-1, keepdim=True)

                    # query_vembeds += vect * 1.
                    to_keep.append(cat_idx)

                elif yn == 'N':
                    query_vembeds -= residual * 0.2
                    # query_vembeds = query_vembeds / query_vembeds.norm(p=2, dim=-1, keepdim=True)

                    # query_vembeds -= vect * 0.2

                    vect = vect.detach().numpy()
                    vect = np.reshape(vect,(1, vect.size))
                    _, _idxs = index.search(vect, 3)
                    _idxs = list(_idxs[0])

                    _cat_idxs = []
                    for idx in _idxs:
                        _cat_idxs.append(category_idxs[idx])

                    to_remove.append(cat_idx)
                    to_remove += _cat_idxs
                else:
                    raise ValueError('yn must be Y or N')

        # Normalize and reform
        query_vembeds = query_vembeds / query_vembeds.norm(p=2, dim=-1, keepdim=True)
        query_vembeds = query_vembeds.detach().numpy()


        print(f'removing {len(to_remove)} products')

        for keep in to_keep:
            if keep in to_remove:
                to_remove.remove(keep)
        # remove nos
        category_idxs = np.setdiff1d(category_idxs, to_remove)
        print(len(category_idxs))

        # Create the index
        category_vembeds = self.embeddings[category_idxs]
        dim = query_vembeds.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(category_vembeds)

        # #### FOR RECOMMENDING OFF INDIVIDUAL LIKES
        # front = []
        # if eval_adj is not None:
        #     to_remove = []
        #     for yn, cat_idx, vect in eval_adj:
        #         if yn == 'Y':
        #             front.append(self.get_product(cat_idx))
        #             to_remove.append(cat_idx)
        #     category_idxs = np.setdiff1d(category_idxs, to_remove)

        # # Search the index
        # if eval_adj is not None:
        #     # for getting products and removing from search
        #     _cat_idxs = []
        #     # where we'll store products per query
        #     products = []

        #     # figure out how many per liked item
        #     n_eval = 0
        #     for yn, _, _ in eval_adj:
        #         if yn == 'Y':
        #             n_eval += 1
        #     n_per = num_results // n_eval

        #     # search for each item
        #     for yn, cat_idx, vect in eval_adj:
        #         if yn == 'Y':
        #             # remove previously found
        #             category_idxs = np.setdiff1d(category_idxs, _cat_idxs)
        #             print(len(category_idxs))
        #             # build index
        #             category_vembeds = self.embeddings[category_idxs]
        #             dim = query_vembeds.shape[1]
        #             index = faiss.IndexFlatIP(dim)
        #             index.add(category_vembeds)

        #             # build search vector
        #             vect = np.reshape(vect,(1, vect.size))
        #             vect = torch.tensor(vect)
        #             vect = vect / vect.norm(p=2, dim=-1, keepdim=True)

        #             vect = vect + query_vembeds * 0.5
        #             vect = vect / vect.norm(p=2, dim=-1, keepdim=True)
        #             vect = vect.detach().numpy()

        #             # search
        #             _, _idxs = index.search(vect, n_per)
        #             _idxs = list(_idxs[0])

        #             # add products
        #             _cat_idxs = []
        #             for idx in _idxs:
        #                 _cat_idxs.append(category_idxs[idx])
        #             _products = []
        #             for cat_idx in _cat_idxs:
        #                 _products.append(self.get_product(cat_idx))
        #             products.append(_products)

        #     # interleave products
        #     out = []
        #     for iprod in range(n_per):
        #         prods = []
        #         for iquery in range(len(products)):
        #             prods.append(products[iquery][iprod])
        #         random.shuffle(prods)
        #         out += prods
        #     products = front + out
        # else:
        #     _, _idxs = index.search(query_vembeds, num_results)
        #     _idxs = list(_idxs[0])
        #     # idxs += _idxs

        #     products = []
        #     for idx in _idxs:
        #         products.append(self.get_product(category_idxs[idx]))


        ##### FOR TRADITIONAL SEARCH
        idxs = []
        _, _idxs = index.search(query_vembeds, max(num_results - len(idxs), 1))
        _idxs = list(_idxs[0])
        idxs += _idxs

        # remove dups
        res = []
        [res.append(x) for x in idxs if x not in res]
        idxs = res

        # Get the products
        products = []
        for idx in idxs:
            products.append(self.get_product(category_idxs[idx], should_load_images))
        return products

    # Function to get the product set type
    def get_type(self) -> str:
        raise NotImplementedError

    # Function to save the product set
    def save(self) -> None:

        # If we're local, then save the file
        if self.data_source == 'local':
            product_set_path = ProductSetFactory.get_local_product_set_path(self.get_type())
            print(f"Saving {self.get_type()} product set to {product_set_path}")

            # Save the pickled product set
            with open(product_set_path, 'wb') as f:
                pickle.dump(self, f)
        
        # Otherwise, save to s3
        elif self.data_source == 's3':
            raise NotImplementedError
        
        # Otherwise, error
        else:
            raise ValueError("Invalid data source")


# ProductSet factory class
class ProductSetFactory:

    # Helper to get the local product set path
    @staticmethod
    def get_local_product_set_path(p_type: str) -> str:
        return os.path.join(get_data_dir(), f"{p_type}_product_set.pkl")

    # Static method to load a saved product set
    @staticmethod
    def load_product_set(p_type: str, data_source: str) -> ProductSet:

        # If we're local, then look for the file path
        if data_source == 'local':
            product_set_path = ProductSetFactory.get_local_product_set_path(p_type)

            # If the file exists, then load it
            if os.path.exists(product_set_path):
                print(f"Loading {p_type} product set from {product_set_path}")
                with open(product_set_path, 'rb') as f:
                    return pickle.load(f)
        
        # Otherwise, check s3
        elif data_source == 's3':

            s3 = get_s3_resource()
            try:
                print(f"Loading {p_type} product set from S3...")
                obj = s3.Object(get_s3_bucket_name(), f"{p_type}/{p_type}_product_set.pkl").get()
                print(f"Unpacking {p_type} product set...")
                return pickle.load(io.BytesIO(obj['Body'].read()))
            except Exception as e:
                pass
        
        # Otherwise, error
        else:
            raise ValueError("Invalid data source")

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

