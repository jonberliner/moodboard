from flask import Flask, render_template, request, jsonify
from PIL import Image
import requests
import hashlib
import sqlite3
import ast
import json
import os
import torch
import datetime
import numpy as np

from gister import Gister
from utils.all_utils import image_to_base64, get_gist_db_path, save_gist_db_to_s3, download_gist_db_from_s3

app = Flask(__name__)


# Endpoint to return the ip
@app.route("/ip", methods=['GET'])
def ip():
    return get_request_ip()

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
                    search_url text, search_url_b text, search_text text, img_weight text, text_weight text, ip text, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    
    # Create a table for all of the search image urls
    c.execute('''CREATE TABLE search_image_urls
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_image_url text)''')

    # Create a table for the evaluations, with fields for the search image url, match_url, evaluation, and ip
    c.execute('''CREATE TABLE evaluations
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_url text, search_url_b text, weight text, match_url text, evaluation text, ip text, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, search_text text, text_weight text)''')

    # Create a table to store a phone number, seed url, a list of urls, a timestamp, and a vector
    c.execute('''CREATE TABLE text_chats
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phone_number text, search_url text, urls text, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, vector ARRAY)''')

    # Seed the search image urls
    search_images = ['https://i.pinimg.com/474x/3f/e7/00/3fe700a9b46de92a7e4b32843ecc2923.jpg',
            'https://i.pinimg.com/736x/ac/13/3f/ac133f87630018fa69fbcddfb61bf2f5.jpg',
            'https://i.pinimg.com/564x/03/62/3c/03623c181de7c18a84932635e0f4279a.jpg',
            'https://i.pinimg.com/474x/fa/73/7b/fa737b8f99636afc99e1de33c3fa70f7.jpg',
            'https://i.pinimg.com/736x/b4/b9/3e/b4b93eeb8a390220c3abdb0c8914f13b.jpg',
            'https://i.pinimg.com/474x/b4/7f/00/b47f00b3cb1aab0efa16ac7f55251757.jpg',
            'https://i.pinimg.com/564x/e7/c7/10/e7c7108847b90e1adc749ce078e06b66.jpg',
            'https://i.pinimg.com/474x/c2/7a/23/c27a23b3e6ba757ccfc5ce899faec376.jpg',
            'https://i.pinimg.com/474x/c9/be/10/c9be10f8ac165a88d1251495ba6cf00d.jpg',
            'https://i.pinimg.com/474x/dc/ad/89/dcad89f8debe0a1e94999959206fb7ab.jpg']
    
    # Create a table for the sms searches

    # Save the search image urls
    for search_image_url in search_images:
        c.execute("INSERT INTO search_image_urls (search_image_url) VALUES (?)", (search_image_url,))

    # Commit and close
    conn.commit()
    conn.close()


# An endpoing to save the database to s3
@app.route("/save_db", methods=['GET'])
def save_db():
    save_gist_db_to_s3()
    return "Saved database to s3"

# An endpoint to download the database
@app.route("/download_db", methods=['GET'])
def download_db():
    download_gist_db_from_s3()
    return "Downloaded database from s3"

def save_request(search_image_url, search_image_url_b, search_ip, search_text, text_weight, img_weight):

    # Get the current datetime
    now = datetime.datetime.now()

    # Connect to the database
    conn = sqlite3.connect(get_gist_db_path())

    # Create a cursor
    c = conn.cursor()   

    # Insert the search url and ip address
    c.execute("INSERT INTO search_history (search_url, search_url_b, search_text, img_weight, text_weight, ip, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)", (search_image_url, search_image_url_b, search_text, img_weight, text_weight, search_ip, now))

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


# Create an endpoint to clear the evaluations
@app.route("/clear_evaluations", methods=['GET'])
def clear_evaluations():

    # Clear all evaluations from the database
    conn = sqlite3.connect(get_gist_db_path())
    c = conn.cursor()
    c.execute("DELETE FROM evaluations")
    conn.commit()
    conn.close()

    # Return a message
    return "Cleared evaluations"


# Endpoint to preload all of the images
@app.route("/preload_images", methods=['GET'])
def preload_images():

    # Load the images
    app.gister.product_set.load_images(preload_all=True)

    # Return the number of products
    return f"Loaded {app.gister.get_num_products()} images"

def get_image_urls():

    # Connect to the database
    conn = sqlite3.connect(get_gist_db_path())

    # Create a cursor
    c = conn.cursor()

    # Get the search history
    c.execute("SELECT * FROM search_image_urls")

    # Get the results
    rows = c.fetchall()

    # Close the connection
    conn.close()

    # Return the results
    return rows
               
# Endpoint to return the search urls
@app.route("/search_urls", methods=['GET','POST'])
def search_urls():

    # See if we're trying to add a url
    add_url = request.form.get("add-url")

    # If we are, add it
    if add_url:
        internal_save_url(add_url)

    # See if we're deleting one
    delete_id = request.form.get("delete-id")

    # If we are, delete it
    if delete_id:
                
        # Connect to the database
        conn = sqlite3.connect(get_gist_db_path())

        # Create a cursor
        c = conn.cursor()

        # Insert the search url and ip address
        c.execute("DELETE FROM search_image_urls WHERE id=?", (delete_id,))

        # Commit and close
        conn.commit()
        conn.close()    

    # Get them
    search_image_urls = get_image_urls()

    # Return the results
    return render_template("search_urls.html", search_image_urls=search_image_urls)


def internal_save_url(url):
        
    # Connect to the database
    conn = sqlite3.connect(get_gist_db_path())

    # Create a cursor
    c = conn.cursor()

    # Insert the search url and ip address
    c.execute("INSERT INTO search_image_urls (search_image_url) VALUES (?)", (url,))

    # Commit and close
    conn.commit()
    conn.close()


# Create an endpoint to save a url
@app.route("/save_url", methods=['POST'])
def save_url():
    
    # Get the url
    save_url = search_image_url = request.form.get("save-url")

    # Save the url
    internal_save_url(save_url)
    
    # Return the url
    return f"Saved {save_url}"

# An endpoint that shows all the evaluations
@app.route("/evaluations", methods=['GET'])
def evaluations():
    
    # Connect to the database
    conn = sqlite3.connect(get_gist_db_path())

    # Create a cursor
    c = conn.cursor()

    # Get the search history
    c.execute("SELECT * FROM evaluations")

    # Get the results
    rows = c.fetchall()

    # Close the connection
    conn.close()

    # Return the results
    return render_template("evaluations.html", rows=rows)

# Create an endpoint to save an evaluation
@app.route("/save_eval", methods=['POST'])
def save_eval():

    # Get the arguments
    eval = request.form.get("eval")
    search_url = request.form.get("search-url")
    search_url_b = request.form.get("search-url-b")
    weight = request.form.get("weight")
    match_url = request.form.get("match-url")
    search_text = request.form.get("search-text")
    if search_text == "":
        search_text = None
    text_weight = request.form.get("text-weight")
    if text_weight == "":
        text_weight = None

    # Get the IP
    ip = get_request_ip()

    # Connect to the database
    conn = sqlite3.connect(get_gist_db_path())

    # Create a cursor
    c = conn.cursor()

    # Insert 
    c.execute("INSERT INTO evaluations (evaluation, search_url, search_url_b, weight, match_url, ip, search_text, text_weight) VALUES (?, ?, ?, ?,?,?,?,?)", (eval, search_url, search_url_b, weight, match_url, ip, search_text, text_weight))

    # Commit and close
    conn.commit()
    conn.close()

    # Return the url
    return f"Saved {eval} for {search_url} and {match_url}"

def get_request_ip():
    # return request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
    return request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)

# Get an embedding vector that reflects the evaluations
def get_eval_adj_embedding(search_image_url):

    # Retrieve all evaluations for this image from the database
    conn = sqlite3.connect(get_gist_db_path())
    c = conn.cursor()
    c.execute("SELECT * FROM evaluations WHERE search_url=?", (search_image_url,))
    rows = c.fetchall()
    conn.close()

    
    # If none, then return None
    if len(rows) == 0:
        return None

    # Create an np zero vector of the right size
    embedding = np.zeros(app.gister.product_set.embeddings.shape[1])

    # Add the evaluations to the vector
    for row in rows:

        # Get the product index
        # TODO: Generalize this
        product_index = app.gister.product_set.image_url_to_idx[row[4]]

        # Get the vector
        vect = app.gister.product_set.embeddings[product_index]

        # Add if positive, subtract if negative
        if row[5] == 'Y':
            embedding += vect
        elif row[5] == 'N':
            embedding -= vect
        
        # Otherwise error
        else:
            raise Exception("Invalid evaluation")

    # Return the embedding
    return embedding

# Create an endpoint to return the closest n images to a given url
@app.route("/get_images", methods=['GET'])
def get_images():

    # Load the gister if we haven't already
    if not hasattr(app, 'gister'):

        # Default to online data
        internal_load_products('asos', 's3', False)
    
    # Get the url from the request
    search_image_url = request.args.get("search_url")

    # Get any text
    search_text = request.args.get("text")

    # Should have either a url or text, but not both
    if search_image_url is None and search_text is None:
        raise Exception("Must provide either a url or text")
    if search_image_url is not None and search_text is not None:
        raise Exception("Must provide either a url or text, but not both")
    
    # Get the phone number from the request
    phone_number = request.args.get("phone_number")

    # Must have a phone number
    if phone_number is None:
        raise Exception("Must provide a phone number")

    # Get the product categories
    categories = app.gister.get_product_categories()
    categories = sorted(categories)

    # Use the first one for now
    category = categories[0]

    # Default to get the first result
    result_type = 'first'

    # If we have a search url, we're starting a new search
    if search_image_url is not None:

        # Remove all text chats with this phone number
        conn = sqlite3.connect(get_gist_db_path())
        c = conn.cursor()
        c.execute("DELETE FROM text_chats WHERE phone_number=?", (phone_number,))
        conn.commit()
        conn.close()

        # Create an np zero vector of the right size and type float32
        np_adj_embedding = np.zeros(app.gister.product_set.embeddings.shape[1], dtype=np.float32)

    # If no image, then use the text and build the vectors
    else:

        # Load the search url and base adj_vector from the most recent text chat that has the same phone number
        conn = sqlite3.connect(get_gist_db_path(), detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()
        c.execute("SELECT * FROM text_chats WHERE phone_number=? ORDER BY id DESC", (phone_number,))
        rows = c.fetchall()
        conn.close()
        
        # Get the search url from the first one
        search_image_url = rows[0][2]

        # Get the images that we sent last time
        last_images = ast.literal_eval(rows[0][3])
        
        # Only expect one image last time
        if len(last_images) != 1:
            raise Exception("Expected one image last time")
        last_image_url = last_images[0]
        
        # Get the old adj vector
        old_adj_vector = rows[0][5]

        # See if we're y or n, then we're updating an existing search
        if search_text.strip().lower() == "y" or search_text.strip().lower() == "n":

            # Get the product index
            # TODO: Generalize this
            product_index = app.gister.product_set.image_url_to_idx[last_image_url]

            # Get the vector
            np_adj_embedding = app.gister.product_set.embeddings[product_index]

            # If 'n', then set the multiplier to -1
            if search_text.strip().lower() == "n":
                np_adj_embedding *= -1

        # Special command to get the next one
        elif search_text.strip().lower() == "f":

            # Not going to adjust
            np_adj_embedding = np.zeros(app.gister.product_set.embeddings.shape[1], dtype=np.float32)

            # But signal to get the next one in the results
            result_type = 'next'

        # Special command to get the last one again
        elif search_text.strip().lower() == "b":

            # Not going to adjust
            np_adj_embedding = np.zeros(app.gister.product_set.embeddings.shape[1], dtype=np.float32)

            # But signal to get the next one in the results
            result_type = 'prev'

        # Otherwise, we're adding to the adj vector
        else:

            # See if the search text begins with -
            multiplier = 1
            if search_text.startswith("-") and len(search_text) > 1:

                # Strip the -
                search_text = search_text[1:]

                # Set the multiplier to -1
                multiplier = -1

            # Get the text vector
            with torch.no_grad():
                text_vembeds = app.gister.texts_to_embeddings([search_text])
                adj_embedding = text_vembeds[0] * 5 * multiplier

            # Convert the adj embedding to np
            np_adj_embedding = adj_embedding.cpu().numpy()

        # And add
        np_adj_embedding += old_adj_vector

    # Get the image data
    s_image = Image.open(requests.get(search_image_url, stream=True).raw)
    
    # Search
    product_results = app.gister.search_images([s_image], category=category, num_results=50,
                                               search_text=None, text_weight=None, 
                                               eval_adj=np_adj_embedding, should_load_images=False)

    # For now, we'll only return 1
    if result_type == 'first':
        product_results = product_results[:1]

    elif result_type == 'next':

        # Run through until we find the last image
        for i, product in enumerate(product_results):
                
                # If we find the last image, then get the next one
                if product.image_url == last_image_url:
    
                    # Get the next one, if there is one
                    if i+1 < len(product_results):
                        product_results = product_results[i+1:i+2]

                    # Otherwise, just return the last one
                    else:
                        product_results = product_results[-1:]

                    # And break
                    break

    elif result_type == 'prev':

        # Run through until we find the last image
        for i, product in enumerate(product_results):
                
                # If we find the last image, then get the previous one
                if product.image_url == last_image_url:

                    # Get the previous one, if there is one
                    if i-1 >= 0:
                        product_results = product_results[i-1:i]
                    
                    # Otherwise, just return the first one again    
                    else:
                        product_results = product_results[:1]
    
                    # And break
                    break

    # Create an array of dicts to return
    results = []
    for product in product_results:
        p_result = {}

        # Get the image url
        p_result['image_url'] = product.image_url
        if not p_result['image_url'].startswith("http"):
            p_result['image_url'] = "https://" + p_result['image_url']

        # And the product urls
        p_result['product_url'] = product.url
        
        # And the label
        p_result['label'] = product.label

        # And add to the results
        results.append(p_result)

    # Get all of the image urls
    image_urls = [product.image_url for product in product_results]

    # Store the number and the results in the text_chats table, including the adj_embedding as a vector
    conn = sqlite3.connect(get_gist_db_path())
    c = conn.cursor()
    c.execute("INSERT INTO text_chats (phone_number, search_url, urls, vector) VALUES (?, ?, ?, ?)", (phone_number, search_image_url, json.dumps(image_urls), np_adj_embedding))
    conn.commit()
    conn.close()

    # Return the results as a json
    return jsonify(results)

def adapt_array(arr):
    return arr.tobytes()

def convert_array(blob):
    return np.frombuffer(blob, dtype=np.float32)

# NOTE: For saving / loading vectors
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)


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

    # Get only the second element of each tuple
    search_image_urls = [url[1] for url in get_image_urls()]

    # If we don't have an image url, use the first one
    if search_image_url is None:
        search_image_url = search_image_urls[0]

    # See if we have a weight
    text_weight = request.form.get("text-weight")
    if text_weight == '':
        text_weight = None
    if text_weight is not None:
        text_weight = float(text_weight)

    # See if we have text
    search_text = request.form.get("search-text")
    if search_text == '':
        search_text = None

    # Save the query, but don't break if it fails
    else:
        try:
            save_request(search_image_url, None, get_request_ip(), search_text, text_weight, None)

        except Exception as e:
            print(f"Error saving request: {e}")
            pass


    # Get the number of results
    num_results = request.form.get("num-results")

    # Make sure we have a number
    if num_results is None:
        num_results = 20
    else:
        num_results = int(num_results)
    
    # Get the image category
    category = request.form.get("image-category")

    # If we don't have a category, use the first one
    if category is None:
        category = categories[0]

    # Get the image data
    s_image = Image.open(requests.get(search_image_url, stream=True).raw)
    
    # And any evaluation adjustment
    # TODO: Put in more robustly (this is for a demo)
    eval_adj = get_eval_adj_embedding(search_image_url)

    # Search
    product_results = app.gister.search_images([s_image], category=category, num_results=num_results,
                                               search_text=search_text, text_weight=text_weight, 
                                               eval_adj=eval_adj)
    
    # Return the images from the products
    result_images = [product.image for product in product_results]

    # And the urls
    urls = [product.url for product in product_results]

    # And the image urls
    result_urls = [product.image_url for product in product_results]

    # Convert to base64
    result_data = [image_to_base64(image) for image in result_images]
    search_data = image_to_base64(s_image)

    # Convert to empty string if we don't have any text
    if search_text is None:
        search_text = ""
    if text_weight is None:
        text_weight = ""

    return render_template("search_image.html", search_image_url=search_image_url, 
                           images=result_data, search_image=search_data, num_results=num_results,
                           categories=categories, category_selected=category, urls=urls, enumerate=enumerate, 
                           search_image_urls=search_image_urls, result_urls=result_urls,
                           search_text=search_text, text_weight=text_weight)

# Add a search image endpoint
@app.route("/search_image_2", methods=['GET', 'POST'])
def search_image_2():

    # Load the gister if we haven't already
    if not hasattr(app, 'gister'):

        # Default to online data
        internal_load_products('asos', 's3', False)

    # Get the image url from the submitted form
    search_image_url = request.form.get("search-url")
    search_image_url_b = request.form.get("search-url-b")

    # Get the product categories
    categories = app.gister.get_product_categories()

    # Sort them alphabetically
    categories = sorted(categories)

    # Get only the second element of each tuple
    search_image_urls = [url[1] for url in get_image_urls()]

    # Get the weight
    weight = request.form.get("weight")

    # Default to .5
    if weight is None:
        weight = .5

    # Make sure we're a number
    weight = float(weight)

    # Save the query, but don't break if it fails
    if search_image_url is not None and search_image_url_b is not None:
        try:
            save_request(search_image_url, search_image_url_b, get_request_ip(), None, None, weight)


        except Exception as e:
            print(f"Error saving request: {e}")
            pass

    # If we don't have an image url, default to the first two
    if search_image_url is None:
        search_image_url = search_image_urls[0]
    if search_image_url_b is None:
        search_image_url_b = search_image_urls[1]

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
    s_image_b = Image.open(requests.get(search_image_url_b, stream=True).raw)
    
    # Search
    product_results = app.gister.search_images([s_image, s_image_b], category=category, num_results=num_results, weight=weight)
    
    # Return the images from the products
    result_images = [product.image for product in product_results]

    # And the urls
    urls = [product.url for product in product_results]

    # And the image urls
    result_urls = [product.image_url for product in product_results]

    # Convert to base64
    result_data = [image_to_base64(image) for image in result_images]
    search_data = image_to_base64(s_image)
    search_data_b = image_to_base64(s_image_b)

    return render_template("search_image_2.html", search_image_url=search_image_url, 
                           images=result_data, search_image=search_data, num_results=num_results,
                           categories=categories, category_selected=category, urls=urls, enumerate=enumerate, 
                           search_image_urls=search_image_urls, result_urls=result_urls, 
                           search_image_url_b=search_image_url_b, search_image_b=search_data_b,
                           weight=weight)

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


# Endpoint to return the version
@app.route("/version", methods=['GET'])
def version():
    return "0.0.13"

if __name__ == "__main__":

    # # For debugging, load the products
    internal_load_products('asos', 'local', preload_all=False)

    # Run on port 80
    app.run(host='0.0.0.0', port=80)