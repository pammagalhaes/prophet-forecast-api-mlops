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
