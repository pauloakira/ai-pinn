# Define the provider
provider "google" {
  project = "<YOUR_PROJECT_ID>"
  region  = "us-central1"   # Adjust to your region
  zone    = "us-central1-a" # Adjust to your zone
}

# Enable necessary APIs
resource "google_project_service" "compute" {
  service = "compute.googleapis.com"
}

resource "google_project_service" "container_registry" {
  service = "containerregistry.googleapis.com"
}

# Define the GPU-enabled instance
resource "google_compute_instance" "gpu_vm" {
  name         = "gpu-instance"
  machine_type = "n1-standard-4"
  zone         = "us-central1-a"  # Adjust to your zone

  boot_disk {
    initialize_params {
      image = "projects/deeplearning-platform-release/global/images/family/common-cu113"
      size  = 200
    }
  }

  metadata = {
    "install-nvidia-driver" = "True"
  }

  network_interface {
    network = "default"
    access_config {
      # This is used to assign a public IP to the instance
    }
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
  }

  # Attach an NVIDIA Tesla K80 GPU
  guest_accelerator {
    type  = "nvidia-tesla-k80"
    count = 1
  }

  # Add scopes to allow access to Google APIs
  service_account {
    scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  # Allow SSH access to the VM
  tags = ["gpu-server"]

  tags = ["http-server"]
}

# Add firewall rule to allow SSH access
resource "google_compute_firewall" "allow_ssh" {
  name    = "allow-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["gpu-server"]
}