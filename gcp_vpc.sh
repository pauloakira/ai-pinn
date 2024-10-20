VM_NAME=neural-vem-vm
VPC_NAME=neural-vem-vpc
SUBNET_NAME=neural-vem-subnet
REGION=us-central1   # Use the region, not the zone
ZONE=us-central1-a
DISK_NAME="neural-vem-persistent-disk-0"

# Create a custom VPC network
gcloud compute networks create $VPC_NAME --subnet-mode=custom

# Create a subnet in the custom VPC network
gcloud compute networks subnets create $SUBNET_NAME \
    --network=$VPC_NAME \
    --region=$REGION \
    --range=10.128.1.0/24

# Create the VM using the existing disk
gcloud compute instances create $VM_NAME \
  --zone=$ZONE \
  --machine-type=n1-standard-4 \
  --subnet=$SUBNET_NAME \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-standard \
  --tags=http-server,https-server

# Configure the firewall to allow SSH, HTTP, and HTTPS
gcloud compute firewall-rules create allow-ssh \
    --network=$VPC_NAME \
    --allow=tcp:22

gcloud compute firewall-rules create allow-http \
    --network=$VPC_NAME \
    --allow=tcp:80

gcloud compute firewall-rules create allow-https \
    --network=$VPC_NAME \
    --allow=tcp:443