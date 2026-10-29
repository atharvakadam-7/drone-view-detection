FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 --retries 10 -r requirements.txt

COPY . .

ENTRYPOINT ["python", "track_video.py"]
CMD ["/videos/drone.mp4", "640"]