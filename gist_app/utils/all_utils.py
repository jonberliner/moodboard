import os
import requests
import base64
import io
import boto3


# Get the filepath to the data directory
def get_data_dir():
    return os.path.join(get_gist_app_path(),'..','data')

def get_gist_db_path():
    return os.path.join(get_gist_app_path(),'gist.db')

# Get the filepath to the gist_app directory
def get_gist_app_path():

    # Get the absolute path of the current directory.
    current_path = os.path.abspath(os.path.dirname(__file__))

    # Split the current path into its components.
    current_path_components = current_path.split(os.sep)

    # Find the index of the news_bot directory.
    gist_app_index = current_path_components.index("gist_app")

    # Join the components up to the news_bot directory.
    gist_app_path = os.sep.join(current_path_components[:gist_app_index + 1])

    # Return the path.
    return gist_app_path

# To replace wget on mac
def py_wget(url, save_as):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(save_as, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    file.flush()
        print("Download completed!")
    else:
        print("Error downloading the file!")

# Convert a PIL image to a base64 encoded string
def image_to_base64(image):

    # Convert to base64
    data = io.BytesIO()
    image.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return encoded_img_data.decode('utf-8')

# Helper to get an s3 resource
def get_s3_resource():

    # Set a session with our credentials
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )

    # Return the s3 resource
    return session.resource('s3')

def get_s3_bucket_name() -> str:
    return 'gist-data'
