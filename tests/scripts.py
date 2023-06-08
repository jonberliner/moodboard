# Testing script 
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gist_app')))

from gister import Gister

# Run this if it's the main file
if __name__ == "__main__":

    # Test loading the products
    g = Gister()
    g.load_products()

    print(f"Loaded {g.get_num_products()} products.")

    # Test search
    product = g.get_product(0)
    category = product.category
    results = g.search_image(product.image, category=category, num_results=5)

    # Now by url
    image_url = "https://i.pinimg.com/564x/2a/da/f7/2adaf77f93508acd3e2d3448768be26b.jpg"
    category = "WomensDresses"
    results = g.search_image_url(image_url, category=category, num_results=5)

    print("Done")