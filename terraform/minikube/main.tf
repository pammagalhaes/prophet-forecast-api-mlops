terraform {
  required_providers {
    kubernetes = { source = "hashicorp/kubernetes" }
  }
}

provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "minikube"
}

variable "replicas" {
  default = 1
}

variable "image" {
  default = "rossmann-api:local"
}

resource "kubernetes_deployment_v1" "api" {
  metadata {
    name = "rossmann-api-tf"
  }
  spec {
    replicas = var.replicas
    selector {
      match_labels = { app = "rossmann-api-tf" }
    }
    template {
      metadata {
        labels = { app = "rossmann-api-tf" }
      }
      spec {
        container {
          name              = "api"
          image             = var.image
          image_pull_policy = "Never"
          port {
            container_port = 8000
          }
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "api" {
  metadata {
    name = "rossmann-api-tf"
  }
  spec {
    selector = { app = "rossmann-api-tf" }
    port {
      port        = 80
      target_port = 8000
    }
    type = "NodePort"
  }
}

output "service_name" {
  value = kubernetes_service_v1.api.metadata[0].name
}
