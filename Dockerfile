FROM python:3.10.14

# Downgrade pip to a version that doesn't cause conflicts
RUN pip install pip==24.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /rvc-docker

# Install Python packages individually
RUN pip install wheel setuptools
RUN pip install numba==0.56.4
RUN pip install numpy==1.23.5
RUN pip install scipy==1.9.3
RUN pip install librosa==0.9.1
RUN pip install fairseq==0.12.2
RUN pip install faiss-cpu==1.7.3
RUN pip install gradio==3.36.1
RUN pip install pyworld>=0.3.2
RUN pip install soundfile>=0.12.1
RUN pip install praat-parselmouth>=0.4.2
RUN pip install httpx==0.23.0
RUN pip install tensorboard
RUN pip install tensorboardX
RUN pip install torchcrepe
RUN pip install onnxruntime
RUN pip install demucs
RUN pip install edge-tts
RUN pip install yt_dlp
RUN pip install omegaconf>=2.0.5
RUN pip install flask
RUN pip install ffmpeg
RUN pip install redis rq
RUN pip install uuid

# Now copy the project files
COPY . /rvc-docker

# Expose the port the app will run on
EXPOSE 7860

# Define the command to run your app
CMD ["python3", "app.py"]