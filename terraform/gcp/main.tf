terraform {
  required_providers {
    google = { source = "hashicorp/google" }
  }
}

variable "project_id" {}
variable "region" { default = "europe-west1" }
variable "image" {
  default = "us-docker.pkg.dev/cloudrun/container/hello"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_cloud_run_v2_service" "api" {
  name     = "rossmann-api"
  location = var.region

  template {
    containers {
      image = var.image
      ports {
        container_port = 8080
      }
      resources {
        limits = { memory = "1Gi", cpu = "1" }
      }
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "url" {
  value = google_cloud_run_v2_service.api.uri
}
