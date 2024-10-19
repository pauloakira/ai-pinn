# VM information (modify as necessary)
VM_NAME=cpu-vm
ZONE=us-central1-a

# Create a VM instance
gcloud compute instances create $VM_NAME \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-standard \
  --tags=http-server,https-server

# Assing the role to access Artifact Registry
gcloud projects add-iam-policy-binding golden-torch-439114-e4 \
  --member="serviceAccount:894352243947-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"

# Authenticate in GCP
gcloud auth login
gcloud auth configure-docker

# Access the VM
gcloud compute ssh $VM_NAME --zone=$ZONE