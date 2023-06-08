from concurrent.futures import ThreadPoolExecutor
import json
import os
from PIL import Image
import requests
from tqdm import tqdm
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from utils.all_utils import py_wget

class AmazonClothingDataset(Dataset):
    """
    dataset found at https://data.world/promptcloud/amazon-fashion-products-2020/file/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson
    downloads dataset images fast with multithreading, and includes 
    category-specific subsetting for fast recommendation over a category
    """
    
    def __init__(self, image_dir: str, records: List[dict]):
        self.image_dir = image_dir
        
        os.makedirs(self.image_dir, exist_ok=True)
        
        # download images and return metadata
        mds = self._thread_run(self._download, records)
        mds = [md for md in mds if md is not None]
    
        self.metadata = pd.DataFrame(mds)

        # Will load the embeddings in here
        self.embeddings = None

    @staticmethod
    def download_records(download_dir, fpath):
        os.makedirs(download_dir, exist_ok=True)
        print(download_dir)
        
        _url = "https://query.data.world/s/pnnl7xgiifupk3sa6odnwr4m7fvmn4?dws=00000"
    
        records_path = py_wget(_url, fpath)
        
        return records_path

    @staticmethod
    def read_records(fpath):
        records = []
        with open(fpath, 'r') as fp:
            for line in fp:
                records.append(json.loads(line))
        return records
    
    def _download(self, record):
        """download a single record"""
        md = None
        try:
            image_url = record["medium"].split("|")[0]
            image_path = os.path.join(self.image_dir, record["uniq_id"] + ".jpg")
            if not os.path.exists(image_path):
                image = Image.open(requests.get(image_url, stream=True).raw)
                image.save(image_path)
                print(image)

            label = f"a catalog photo of {record['product_name']}"

            category = list(record["parent___child_category__all"].keys())[1]

            md = {
                "product_id": record["uniq_id"],
                "image_path": image_path,
                "image_url": image_url,
                "label": label,
                "category": category,
            }
        except:
            md = None

        return md

    @staticmethod
    def _thread_run(f, my_iter):
        """download all records using multithreading"""
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
        return results

        
    def __getitem__(self, idx: int) -> (Image, str, str):
        md = self.metadata.iloc[idx]
        
        try:
            image = Image.open(md.image_path)
        except:
            image = None

        return (image, md.label, md.category)
    
    def __len__(self,) -> int:
        return len(self.metadata)
    
    def subset_by_category(self, category: str):
        """return a dataset of only products in this category"""
        records = []
        for i, record in enumerate(self.records):
            rcategory = list(record["parent___child_category__all"].keys())[1]
            
            if rcategory == category:
                records.append(record)

        return AmazonClothingDataset(image_dir=self.image_dir, records=records)
    
    def get_category_indices(self, category: str) -> np.array:
        """return the indices of all products in this category"""
        md = self.metadata[self.metadata.category == category]
        return md.index
    
    # Get the data_loader helper for the embeddings
    def get_data_loader(self, batch_size: int = 32) -> DataLoader:

        def collate_fn(batch):
            images, labels, categories = [], [], []
            for item in batch:
                image, label, category = item
                # filter out any datapoints with corrupt images
                if image is not None:
                    images.append(image)
                    labels.append(label)
                    categories.append(category)
            return images, labels, categories

        # Create the data loader
        data_loader = DataLoader(
            dataset = self,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        return data_loader
