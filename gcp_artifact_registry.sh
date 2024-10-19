PROJECT_ID=golden-torch-439114
IMAGE_NAME=neural-vem

# Authenticate with Google Cloud CLI for Container Registry
gcloud auth configure-docker

# Build the Docker image
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

# Push the image to GCP Container Registry
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

# Clean unused containers, images and volumns
docker system prune -a --volumes