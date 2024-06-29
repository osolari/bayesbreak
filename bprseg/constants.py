import os


REPO_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

# data_resources where public data is kept
DATA_RESOURCES_DIR = os.path.join(REPO_DIR, "external_resources/data")

# Test directory
TEST_DIR = f"{REPO_DIR}/bprseg/test/"
