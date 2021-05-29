import os

PROJECT_NAME = "diagnose-alzheimer"
PROJECT_ROOT_DIR = os.getcwd()[:os.getcwd().rindex(PROJECT_NAME) + len(PROJECT_NAME)]

ADNI_RAW_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "raw", "adni")
ADNI_TRANSFORMED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "transformed", "adni")
ADNI_PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "processed", "adni")