# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11-slim-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# install VIM
RUN apt-get update \
    && apt-get install -y \
        vim

# To fix psutil error
RUN apt-get install -y gcc python3-dev
# RUN apk add build-base linux-headers


# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app

# Copy the application's source code to the /app directory
COPY gist_app /app/gist_app
#COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# Expose the default port
EXPOSE 80

# Create a command to run the flask app
CMD ["python", "gist_app/app.py"]

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD ["python", "web3/web2py.py", "--no_gui", "-a", "admin2", "-i", "0.0.0.0"]
