# Rossmann Prophet Forecast API

This project is a **refactored and improved version** of my first Machine Learning solution for Rossmann Sales Forecasting.  
In this new version, I applied **advanced techniques**, **clean code modularization**, **Dockerization**, and **MLOps practices**, including CI/CD with GitHub Actions and deployment on Render.

The final result is a fully functional **forecasting API** capable of predicting daily sales for Rossmann stores using **Prophet** models.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Project Structure](#project-structure)  
4. [Modeling](#modeling)  
5. [API](#api)  
6. [Docker & Deployment](#docker--deployment)  
7. [GitHub Actions & CI/CD](#github-actions--cicd)  
8. [Usage](#usage)  
9. [Screenshots](#screenshots)  
10. [Future Improvements](#future-improvements)  

---

## Project Overview

The goal of this project is to build a **time series forecasting API** capable of predicting future sales for individual Rossmann stores.

This version brings:

- Cleaner and fully modular code  
- Reusable modeling functions  
- Separation between API, configuration, and modeling logic  
- Environment variable control for Render  
- Docker containerization  
- CI/CD automation with GitHub Actions  
- Deployment on Render  
- Organized model storage  

This allows anyone (or any system) to request forecasts programmatically through a REST API.

---

## Dataset

The original Rossmann dataset contains:

- **`train.csv`** – historical daily sales  
- **`test.csv`** – test set  
- **`store.csv`** – store metadata  

A custom train/validation split was used:  

- **Training until:** `2015-06-01`  
- **Validation after:** that date  

---

## Project Structure

```plaintext
rossmann_prophet/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── prophet_store_X.joblib
├── src/
│   ├── api/
│   │   └── main.py
│   ├── modeling/
│   │   └── model_utils.py
│   └── config.py
├── Dockerfile
├── requirements.txt
└── README.md

---
## Modeling

The forecasting solution uses Prophet with additional regressors:

-Promo

-StateHoliday

-SchoolHoliday

A separate model is trained for each store.
Models are saved into the models/ directory as .joblib files, and the API loads the correct model dynamically based on the input store ID.

## API

The API was built using FastAPI and exposes 2 endpoints:

1. GET /

Health check endpoint
Returns:

{"status": "ok"}

2. POST /predict

Returns future sales forecasts.

📌 IMPORTANT:
To test the API in the browser, use the interactive docs:

➡️ https://prophet-forecast-api-mlops.onrender.com/docs

Example Request
{
  "store_id": 1,
  "periods": 7,
  "Promo": 1,
  "StateHoliday": 0,
  "SchoolHoliday": 0
}

Example Response
{
  "store_id": 1,
  "periods": 7,
  "predictions": [
    {"ds": "2015-05-31T00:00:00", "yhat": 4352.69},
    {"ds": "2015-06-01T00:00:00", "yhat": 4210.15},
    ...
  ]
}

## Docker & Deployment

This project is fully containerized using Docker.

Build locally:
docker build -t rossmann-api .

Run:
docker run -p 8000:8000 rossmann-api

Deployment on Render

The API is deployed at:

➡️ https://prophet-forecast-api-mlops.onrender.com

## GitHub Actions & CI/CD

The project includes GitHub Actions workflows for:

Running unit tests (if added later)

Validating code structure

Building the Docker image

Preparing deployment steps

📸 Screenshot placeholder:
Add an image showing the “green check” workflow run in GitHub Actions.
What should appear:

Pipeline name

Status “Success”

Timestamp

Usage
Test via API Docs

Open:

➡️ https://prophet-forecast-api-mlops.onrender.com/docs

Inside the /predict section:

Click Try it out

Enter input JSON

Click Execute

View the forecast results

📸 Screenshot placeholder:
Include a screenshot of the Swagger UI with the POST /predict executed.
What should appear:

The request JSON you sent

The response JSON from the API

Status code 200

The models predicting without errors

Screenshots
1. Local API Running

📸 Placeholder:
Screenshot of FastAPI running locally at http://localhost:8000/docs.

2. Render Deployment Panel

📸 Placeholder:
Screenshot of Render dashboard showing:

Service name

“Deploy succeeded”

CPU and RAM usage

Logs without errors

3. API Prediction Example

📸 Placeholder:
Screenshot showing the prediction output inside Swagger UI.

## Future Improvements

Add a web interface (Streamlit or Gradio)

Add automated model retraining pipeline

Store models using Git LFS

Add authentication (API tokens)

Create monitoring dashboard (Prometheus + Grafana)

Deploy using Terraform or Pulumi

Add more regressors (weather, competitors, etc.)

Generate forecasts for all stores automatically


