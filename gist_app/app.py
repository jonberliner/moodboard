from flask import Flask, render_template, request
from PIL import Image
import requests
import hashlib
import sqlite3
import os
import datetime

from gister import Gister
from utils.all_utils import image_to_base64, get_gist_db_path

app = Flask(__name__)


# Endpoint to return the version
@app.route("/version", methods=['GET'])
def version():
    return "0.0.5"

# Endpoint to load the products
@app.route("/load_products", methods=['GET'])
def load_products():
    
    # Get the product type and data source
    product_type = request.args.get("product_type")

    # Default to asos
    if product_type is None:
        product_type = "asos"

    data_source = request.args.get("data_source")

    # Default to s3
    if data_source is None:
        data_source = "s3"

    # See if we're using the prebuilt version
    use_prebuilt = request.args.get("use_prebuilt")

    # Convert to boolean
    if use_prebuilt is not None:
        use_prebuilt = use_prebuilt.lower() == "true"

    # Default to true
    else:
        use_prebuilt = True

    # Load the products
    internal_load_products(product_type, data_source, preload_all=False, use_prebuilt=use_prebuilt)

    # Return the number of products
    return f"Loaded {app.gister.get_num_products()} products"

# Create a function to create the gist database
def create_gist_db():

    # Get the path
    db_path = get_gist_db_path()

    # Delete the database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create the database
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create the table with fields for the search url, ip, and time
    c.execute('''CREATE TABLE search_history
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_url text, ip text, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    
    # Commit and close
    conn.commit()
    conn.close()


def save_request(search_image_url, search_ip):

    # Get the current datetime
    now = datetime.datetime.now()

    # Connect to the database
    conn = sqlite3.connect(get_gist_db_path())

    # Create a cursor
    c = conn.cursor()   

    # Insert the search url and ip address
    c.execute("INSERT INTO search_history (search_url, ip, timestamp) VALUES (?, ?, ?)", (search_image_url, search_ip, now))

    # Commit and close
    conn.commit()
    conn.close()

# Endpoint to return the search history
@app.route("/search_history", methods=['GET'])
def search_history():
    
    # Connect to the database
    conn = sqlite3.connect(get_gist_db_path())

    # Create a cursor
    c = conn.cursor()

    # Get the search history
    c.execute("SELECT * FROM search_history")

    # Get the results
    rows = c.fetchall()

    # Close the connection
    conn.close()

    # Return the results
    return render_template("search_history.html", rows=rows)  


def internal_load_products(product_type, data_source, preload_all, use_prebuilt=False):

    # Set up the gister
    with app.app_context():

        g = Gister()
        g.load_product_set(product_type, data_source, preload_all, use_prebuilt)

        # And store for later
        app.gister = g

        # And create a local sqlite database
        create_gist_db()        


# Endpoint to preload all of the images
@app.route("/preload_images", methods=['GET'])
def preload_images():

    # Load the images
    app.gister.product_set.load_images(preload_all=True)

    # Return the number of products
    return f"Loaded {app.gister.get_num_products()} images"


# Add a search image endpoint
@app.route("/search_image", methods=['GET', 'POST'])
def search_image():

    # Load the gister if we haven't already
    if not hasattr(app, 'gister'):

        # Default to online data
        internal_load_products('asos', 's3', False)

    # Get the image url from the submitted form
    search_image_url = request.form.get("search-url")

    # Get the product categories
    categories = app.gister.get_product_categories()

    # Sort them alphabetically
    categories = sorted(categories)

    # If we don't have an image url, return the search page
    if search_image_url is None:
        return render_template("search_image.html", search_image_url='', num_results=20, categories=categories,
                               category_selected="WomensDresses")

    # Save the query, but don't break if it fails
    try:
        save_request(search_image_url, request.environ.get('HTTP_X_REAL_IP', request.remote_addr))

    except Exception as e:
        print(f"Error saving request: {e}")
        pass


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

    # And the urls
    result_urls = [product.url for product in product_results]

    # Convert to base64
    result_data = [image_to_base64(image) for image in result_images]
    search_data = image_to_base64(s_image)

    return render_template("search_image.html", search_image_url=search_image_url, 
                           images=result_data, search_image=search_data, num_results=num_results,
                           categories=categories, category_selected=category, urls=result_urls, enumerate=enumerate)

# Create a route to receive ebay closure notifications
@app.route("/ebay_notify", methods=['GET', 'POST'])
def ebay_notify():

    # Get the challenge_code parameter from the request
    challenge_code = request.args.get("challenge_code")

    # If we have a challenge code, return the response
    if challenge_code is not None:

        # Create a verification token
        verificationToken = "GistApp_76DOUGLAS_2023_1_1_JONATHAN_1"

        # And the endpoint
        endpoint = 'https://gist.possibleworldsconsulting.com/ebay_notify'

        # Create the hash
        m = hashlib.sha256(str(challenge_code+verificationToken+endpoint).encode('utf-8'));

        # Return the hash in json format
        return {"challengeResponse": m.hexdigest()}

    # Otherwise, return a 200
    else:
        return "OK"


@app.route("/")
# Note: This is used as the health check endpoint for the load balancer
def hello():
    return render_template("index.html")


if __name__ == "__main__":

    # # # For debugging, load the products
    # internal_load_products('asos', 'local', preload_all=False)

    # Run on port 80
    app.run(host='0.0.0.0', port=80)