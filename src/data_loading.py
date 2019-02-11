from config import raw_test_data_path, raw_train_data_path, raw_dev_data_path, data_path, test_data_path, dev_data_path, train_data_path
import os
import json


# @param data_path {string}  the path of the folder that contains the data files.
# @param sentiment {int} -1: undefined, 0: negative, 1: positive
def load_raw_data(raw_data_path, sentiment = -1, data = {'file_names':[], 'target':[], 'data':[]}):
    files = os.listdir(raw_data_path)
    for file in files:
        if file[-4:] == '.txt':
            with open(os.path.join(raw_data_path, file), 'r', encoding='utf8') as f:
                print("PROCESSING " + raw_data_path + file)
                data['file_names'].append(file[:-4])
                data['target'].append(sentiment)
                data['data'].append(f.read())
    return data


def write_processed_data(data, outfile_path):
    with open(outfile_path, 'w') as outfile:
        json.dump(data, outfile)


def init():
    train_data = load_raw_data(raw_train_data_path + "/neg/", sentiment=0, data={'file_names': [], 'target': [], 'data': []})
    write_processed_data(load_raw_data(raw_train_data_path+"/pos/", sentiment=1, data=train_data), train_data_path)

    write_processed_data(load_raw_data(raw_test_data_path, sentiment=-1, data={'file_names': [], 'target': [], 'data': []}), test_data_path)
    write_processed_data(load_raw_data(raw_dev_data_path, sentiment=-1, data={'file_names': [], 'target': [], 'data': []}), dev_data_path)


def load_data(path):
    return json.load(open(path, 'r'))




# EXECUTE init() only if data/processed_data/ folder is empty
# init()


# load processed data like this.
# data = load_data(dev_data_path)

