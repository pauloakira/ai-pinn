PROJECT_ID=golden-torch-439114
IMAGE_NAME=neural-vem

# Clean unused containers, images and volumns
docker system prune -a --volumes

# Authenticate with Google Cloud CLI for Container Registry
gcloud auth configure-docker

# Build the Docker image
docker build -t gcr.io/golden-torch-439114-e4/neural-vem:latest .

# Push the image to GCP Container Registry
docker push gcr.io/golden-torch-439114-e4/neural-vem:latest