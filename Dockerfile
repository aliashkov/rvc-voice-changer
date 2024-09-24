FROM python:3.10.14

# Downgrade pip to a version that doesn't cause conflicts
RUN pip install pip==24.0

WORKDIR /rvc-docker

# Copy only requirements.txt first, to cache dependencies
COPY requirements.txt /rvc-docker/

# Install dependencies
RUN pip install -r requirements.txt

# Now copy the rest of the project files
COPY . /rvc-docker

# Expose the port the app will run on
EXPOSE 7860

# Define the command to run your app
CMD ["python3", "app.py"]