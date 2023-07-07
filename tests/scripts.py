# Testing script 
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gist_app')))

from gister import Gister
from asos_product_set import AsosProductSet

from dotenv import load_dotenv

# Run this if it's the main file
if __name__ == "__main__":

    # # # Load the dotenv file
    # load_dotenv(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), '.env'))

    # os.getenv('RAPIDAPI_KEY')

    # # Download the asos categories
    # # category_data = AsosProductSet.download_category_list()
    # product_data = AsosProductSet.download_product_details("204103436")
    # # products_data = AsosProductSet.download_products("8799") # Dresses


    # Test loading the products
    g = Gister()
    # g.load_product_set('amazon')
    g.load_product_set('asos', 'local', preload_all=False)
    print(f"Loaded {g.get_num_products()} products.")

    # Get the product categories
    categories = g.get_product_categories()
    print(f"Number of categories: {len(categories)}")

    # Test search
    product = g.product_set.get_product(10)
    product = g.product_set.get_product(0)
    category = product.category
    results = g.search_image(product.image, category=category, num_results=5)

    # Now by url
    image_url = "https://i.pinimg.com/564x/2a/da/f7/2adaf77f93508acd3e2d3448768be26b.jpg"
    category = "WomensDresses"
    results = g.search_image_url(image_url, category=category, num_results=5)

    print("Done")