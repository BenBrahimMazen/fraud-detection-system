from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_RAW_PATH      = ROOT_DIR / os.getenv("DATA_RAW_PATH", "data/raw/creditcard.csv")
DATA_PROCESSED_DIR = ROOT_DIR / os.getenv("DATA_PROCESSED_PATH", "data/processed")
DATA_INTERIM_DIR   = ROOT_DIR / os.getenv("DATA_INTERIM_PATH", "data/interim")
MODEL_PATH         = ROOT_DIR / os.getenv("MODEL_PATH", "data/processed/best_model.pkl")

MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")

API_HOST     = os.getenv("API_HOST", "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT", 8000))
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

RANDOM_STATE    = int(os.getenv("RANDOM_STATE", 42))
TEST_SIZE       = float(os.getenv("TEST_SIZE", 0.2))
CV_FOLDS        = int(os.getenv("CV_FOLDS", 5))
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", 0.5))

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = ["Time", "Amount"]
TARGET       = "Class"
ALL_FEATURES = PCA_FEATURES + RAW_FEATURES