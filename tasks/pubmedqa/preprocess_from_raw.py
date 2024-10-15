import os
import json

raw_data_dir = "data/raw/pubmedqa/data/"
data_dir = "data/"

combined_train_data = {}

# train and dev data
for root, dirs, files in os.walk(raw_data_dir):
    if os.path.basename(root).startswith('pqal_fold'):
        train_set_path = os.path.join(root, 'train_set.json')
        
        if os.path.exists(train_set_path):
            with open(train_set_path, 'r') as json_file:
                data = json.load(json_file)
                combined_train_data.update(data)
        

print("Combined train data keys:", len(combined_train_data))
#json.dump(combined_train_data, open(f"{data_dir}/train_set.json", 'w'))
#save in beautiful json format
json.dump(combined_train_data, open(f"{data_dir}/train_set.json", 'w'), indent=4)

test_data = json.load(open(f"{raw_data_dir}test_set.json", 'r'))
print("Test data keys:", len(test_data))
json.dump(test_data, open(f"{data_dir}/test_set.json", 'w'), indent=4)







