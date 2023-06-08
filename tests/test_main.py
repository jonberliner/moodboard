# Make sure we have the right path for testing
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','gist_app')))

from gister import Gister

# Test loading the products
def test_load_products():

    g = Gister()
    g.load_products()

    # Make sure we have the right number of products
    assert g.get_num_products() == 24776

    # And that we loaded the embeddings
    assert g.get_num_product_embeddings() == g.get_num_products()

# Test search
def test_search():

    g = Gister()
    g.load_products()

    # Test search
    product = g.get_product(0)
    category = product.category
    num_results = 5
    results = g.search(product.image, category=category, num_results=num_results)

    # Make sure we have the right number of results
    assert len(results) == num_results

    # Make sure the image, label, and category are the same for the first result and the product we searched for
    assert results[0].image == product.image
    assert results[0].label == product.label
    assert results[0].category == product.category

# Test search by url
def test_search_by_url():

    g = Gister()
    g.load_products()

    image_url = "https://i.pinimg.com/564x/2a/da/f7/2adaf77f93508acd3e2d3448768be26b.jpg"
    category = "WomensDresses"
    results = g.search_image_url(image_url, category=category, num_results=5)

    # Make sure we have the right number of results
    assert len(results) == 5

    # Make sure the image, label, and category are the same for the first result and the product we searched for
    assert results[0].category == category

    # TODO: Eventually set objective correctness criteria for the results