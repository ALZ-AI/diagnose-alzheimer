import os

PROJECT_NAME = "diagnose-alzheimer"
PROJECT_ROOT_DIR = os.getcwd()[:os.getcwd().index(PROJECT_NAME) + len(PROJECT_NAME)]

TRAIN_RAW_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "raw", "alzheimer", "train")
TEST_RAW_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "raw", "alzheimer", "test")

TRAIN_PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "processed", "alzheimer", "train")
TEST_PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "processed", "alzheimer", "test")