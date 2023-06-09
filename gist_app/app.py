from flask import Flask, render_template, request
from PIL import Image
import requests

from gister import Gister
from utils.all_utils import image_to_base64

app = Flask(__name__)

# Add a search image endpoint
@app.route("/search_image", methods=['GET', 'POST'])
def search_image():

    # Get the image url from the submitted form
    search_image_url = request.form.get("search-url")

    # Get the product categories
    categories = app.gister.get_product_categories()

    # Sort them alphabetically
    categories = sorted(categories)

    # If we don't have an image url, return the search page
    if search_image_url is None:
        return render_template("search_image.html", search_image_url='', num_results=5, categories=categories,
                               category_selected="WomensDresses")

    # Get the number of results
    num_results = request.form.get("num-results")

    # Make sure we have a number
    if num_results is None:
        num_results = 5
    else:
        num_results = int(num_results)
    
    # Get the image category
    category = request.form.get("image-category")

    # If we don't have a category, use the first one
    if category is None:
        category = categories[0]

    # Get the image data
    s_image = Image.open(requests.get(search_image_url, stream=True).raw)
    
    # Search
    product_results = app.gister.search_image(s_image, category=category, num_results=num_results)
    
    # Return the images from the products
    result_images = [product.image for product in product_results]

    # Convert to base64
    result_data = [image_to_base64(image) for image in result_images]
    search_data = image_to_base64(s_image)

    return render_template("search_image.html", search_image_url=search_image_url, 
                           images=result_data, search_image=search_data, num_results=num_results,
                           categories=categories, category_selected=category)


@app.route("/")
def hello():
  return str(app.gister.get_num_products())
  return "Hello World!"

if __name__ == "__main__":
  
    with app.app_context():
        g = Gister()
        g.load_products()

        # And store for later
        app.gister = g

    app.run()