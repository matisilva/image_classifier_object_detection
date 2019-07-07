import json
import os
from tqdm import tqdm


anns_dir = "./train_annotations_json"
anns_dir_2 = "./train_annotations_json_2"
categories = os.listdir(anns_dir)
for category in tqdm(categories):
    with open(os.path.join(anns_dir, category)) as f:
        anns = json.load(f)
    for file_id in list(anns.keys())[:30]:
        category_n = category.split("annotations_json_")[1].split('.')[0]
        with open(os.path.join(anns_dir_2, category_n, file_id + '.json')) as f:
            new_anns = json.load(f)
        if new_anns['detections'] != anns[file_id]:
            print(anns[file_id])
            print(new_anns['detections'])
            input()
