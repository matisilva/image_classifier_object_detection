from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import xml.etree.ElementTree as ET
import os
import sys
import pickle
import random


ANNOTATIONS_DIR = "./train_annotations_instance"


def get_vocabulary():
    files = os.listdir(ANNOTATIONS_DIR)
    print("Getting vocabulary words")
    features = set()
    for file in tqdm(files):
        file = os.path.join(ANNOTATIONS_DIR, file)
        root = ET.parse(file).getroot()
        for type_tag in root.findall('object'):
            value = type_tag.find('name')
            features.add(value.text.strip())
    return list(features)


def parse_dataset(vectorizer=None):
    mode = 'test'
    if vectorizer is None:
        mode = 'train'
    files = os.listdir(ANNOTATIONS_DIR)
    # vocabulary = get_vocabulary()
    print("Making training matrix")
    featurized_imgs = []
    matrix = []
    labels = []
    if mode == 'train':
        vectorizer = DictVectorizer(sparse=False)
    for file in tqdm(files):
        file = os.path.join(ANNOTATIONS_DIR, file)
        root = ET.parse(file).getroot()
        featurized_img = defaultdict(int)
        for type_tag in root.findall('object'):
            value = type_tag.find('name')
            featurized_img[value.text.strip()] += 1
        class_img = random.randint(0, 10)  # WARNING: DO THIS before training
        featurized_imgs.append(featurized_img)
        labels.append(class_img)
    if mode == 'train':
        matrix = vectorizer.fit_transform(featurized_imgs)
    else:
        matrix = vectorizer.transform(featurized_imgs)
    return matrix, labels, vectorizer


def make_model(model_type):
    if model_type == 'svm':
        from sklearn import svm
        clf = svm.SVC(gamma='scale')
    elif model_type == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                     random_state=0)
    else:
        return None
    return clf


def train(feature_matrix, labels, model_type='svm', save=True):
    clf = make_model(model_type)
    if not clf:
        raise Exception("No clf setted")
    print('Model fitting')
    clf.fit(feature_matrix, labels)
    return clf


def test(feature_matrix, labels, clf):
    labels = clf.predict(feature_matrix)
    return labels


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("BAD ARGS")
    if sys.argv[1] == 'train':
        print("TRAIN")
        feature_matrix, labels, vectorizer = parse_dataset()
        print("Vectorizer ok")
        clf = train(feature_matrix, labels)
        print("Model ok")
        _now = str(datetime.now())
        filename = "clf-{}-{}.pickle".format(clf.__class__.__name__, _now)
        pickle.dump(clf, open(filename, 'wb'))
        filename = "vec-{}-{}.pickle".format(clf.__class__.__name__, _now)
        pickle.dump(vectorizer, open(filename, 'wb'))
        print("Model and vectorizer saved ok")
    if sys.argv[1] == 'test':
        print("TEST")
        loaded_model = pickle.load(open(sys.argv[2], 'rb'))
        vectorizer = pickle.load(open(sys.argv[3], 'rb'))
        feature_matrix, labels, vectorizer = parse_dataset(vectorizer)
        labels = test(loaded_model)
        print(labels)
