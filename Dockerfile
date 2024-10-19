# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the code and requirements
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Set the environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Run the script
CMD ["python", "main.py"]