import os.path

# Environment variables

raw_data_path = os.path.dirname(__file__) + "/data/comp-551-imbd-sentiment-classification/"
raw_test_data_path = raw_data_path + "test/test/"
raw_train_data_path = raw_data_path + "train/train/"
raw_dev_data_path = raw_data_path + "dev/"      # tiny data set for testing during development

data_path = os.path.dirname(__file__) + "/data/processed_data/"
test_data_path = data_path + "test.json"
train_data_path = data_path + "train.json"
dev_data_path = data_path + "dev.json"

