import os
from tqdm import tqdm
from random import shuffle

BIG_TRAIN_FOLDER = './train_annotations_json_2'
SAMPLE_TRAIN_FOLDER = './sample_annotations_json_2'

os.mkdir(SAMPLE_TRAIN_FOLDER)
for category in tqdm(os.listdir(BIG_TRAIN_FOLDER)):
    if category.startswith('.'):
        continue
    try:
        os.mkdir(os.path.join(SAMPLE_TRAIN_FOLDER, category))
    except Exception:
        pass
    files = os.listdir(os.path.join(BIG_TRAIN_FOLDER, category))
    files = list(files)
    shuffle(files)
    files = files[:int(.3 * len(files))]
    for file in files:
        from_file = os.path.join(BIG_TRAIN_FOLDER, category, file)
        to_file = os.path.join(SAMPLE_TRAIN_FOLDER, category, file)
        os.system('cp {} {}'.format(from_file, to_file))
