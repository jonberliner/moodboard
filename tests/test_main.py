# Make sure we have the right path for testing
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','gist_app')))

from gister import Gister

# Test loading the products
def test_load_products():

    # The number of products we expect
    product_sets = {'amazon':[24776,217,'local'],'asos':[19545,1,'local'],'asos':[19545,1,'s3']}

    # Try for both amazon and asos
    for product_set, args in product_sets.items():

        # Divide the args
        num_products, num_categories, data_source = args

        g = Gister()
        g.load_product_set(product_set, data_source)

        # Make sure we have the right number of products
        assert g.get_num_products() == num_products

        # And that we loaded the embeddings
        assert g.get_num_product_embeddings() == g.get_num_products()

        # And that we can get the categories
        assert len(g.get_product_categories()) == num_categories


# Test search
def test_search():

    # Test for all the product set types
    for product_set in ['amazon','asos']:

        g = Gister()
        g.load_product_set(product_set)

        # Test search
        product = g.product_set.get_product(0)
        category = product.category
        num_results = 5
        results = g.search_image(product.image, category=category, num_results=num_results)

        # Make sure we have the right number of results
        assert len(results) == num_results

        # Make sure the image, label, and category are the same for the first result and the product we searched for
        assert results[0].image == product.image
        assert results[0].label == product.label
        assert results[0].category == product.category

# Test search by url
def test_search_by_url():

    # Test for all the product set types
    for product_set in ['amazon','asos']:

        g = Gister()
        g.load_product_set(product_set)

        image_url = "https://i.pinimg.com/564x/2a/da/f7/2adaf77f93508acd3e2d3448768be26b.jpg"
        category = "WomensDresses"
        results = g.search_image_url(image_url, category=category, num_results=5)

        # Make sure we have the right number of results
        assert len(results) == 5

        # Make sure the image, label, and category are the same for the first result and the product we searched for
        assert results[0].category == category

        # TODO: Eventually set objective correctness criteria for the results
