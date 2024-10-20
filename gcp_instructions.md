## Access the VM

After the VM creation, to access it using SSH:

```
gcloud compute ssh <your-vm-name> --zone=<your-zone>
```

Turn on the VM:

```
gcloud compute instances start cpu-vm --zone=us-central1-a
```

Turn off the VM:

```
gcloud compute instances stop cpu-vm --zone=us-central1-a
```

## Docker + VM

Inside the VM, we need to install docker:

```
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

To pull the docker image:

```
docker pull gcr.io/<your-project-id>/<your-image-name>:<tag>
```

To run the container:

```
docker pull gcr.io/<your-project-id>/<your-image-name>:<tag>
```

For example:

```
sudo docker pull gcr.io/golden-torch-439114-e4/neural-vem:latest

sudo docker run -it --rm \
  --name neural-vem-container \
  gcr.io/golden-torch-439114-e4/neural-vem:latest
```
