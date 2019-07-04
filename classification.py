import xml.etree.ElementTree as ET
import os
import sys
import pickle
import csv
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


TRAIN_DIR = "./train_annotations_instance"
TEST_DIR = "./test_annotations_instance"
LABELS_FILE = "./train.csv"


def get_vocabulary(annotations_dir=TRAIN_DIR):
    files = os.listdir(annotations_dir)
    print("Getting vocabulary words")
    features = set()
    for file in tqdm(files):
        file = os.path.join(annotations_dir, file)
        root = ET.parse(file).getroot()
        for type_tag in root.findall('object'):
            value = type_tag.find('name')
            features.add(value.text.strip())
    return list(features)


def parse_annotated_labels():
    annotated_labels = dict()
    print("Getting labels from annotations")
    with open(LABELS_FILE) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in tqdm(csvReader):
            annotated_labels[row[1]] = row[2]
    return annotated_labels


def parse_dataset(annotations_dir=TRAIN_DIR, vectorizer=None):
    mode = 'test'
    if vectorizer is None:
        mode = 'train'
    files = os.listdir(annotations_dir)
    # vocabulary = get_vocabulary()
    print("Making training matrix")
    featurized_imgs = []
    matrix = []
    labels = []
    if mode == 'train':
        vectorizer = DictVectorizer(sparse=False)
        annotated_labels = parse_annotated_labels()
    for file in tqdm(files):
        file_id = file.split(".")[0]
        file = os.path.join(annotations_dir, file)
        root = ET.parse(file).getroot()
        featurized_img = defaultdict(int)
        for type_tag in root.findall('object'):
            value = type_tag.find('name')
            featurized_img[value.text.strip()] += 1
        featurized_imgs.append(featurized_img)
        if mode == 'train':
            class_img = annotated_labels[file_id]
            labels.append(class_img)
    if mode == 'train':
        matrix = vectorizer.fit_transform(featurized_imgs)
    else:
        matrix = vectorizer.transform(featurized_imgs)
    return matrix, labels, vectorizer


def make_model(model_type=None):
    clf1 = LogisticRegression(solver='lbfgs',
                              multi_class='multinomial',
                              random_state=1,
                              max_iter=400)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    clf4 = svm.SVC(gamma='scale', probability=True)
    estimators = [('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svm', clf4)]
    eclf = VotingClassifier(estimators=estimators, voting='soft',
                            weights=[1, 1, 1, 1], flatten_transform=True)
    if model_type == 'lr':
        return clf1
    if model_type == 'rf':
        return clf2
    if model_type == 'gnb':
        return clf3
    if model_type == 'svm':
        return clf4
    return eclf


def train(feature_matrix, labels, model_type=None, save=True):
    clf = make_model(model_type)
    if not clf:
        raise Exception("No clf setted")
    print('Model fitting')
    clf.fit(feature_matrix, labels)
    return clf


def eval(feature_matrix, labels, model_type=None, save=True):
    clf = make_model(model_type)
    print(clf.__class__.__name__)
    if not clf:
        raise Exception("No clf setted")
    print('Model evaluation')
    train_m, test_m, train_labels, test_labels = train_test_split(
        feature_matrix, labels, test_size=0.33, random_state=42)
    clf.fit(train_m, train_labels)
    pred_labels = clf.predict(test_m)
    print(classification_report(test_labels, pred_labels))
    print(balanced_accuracy_score(test_labels, pred_labels))
    return clf


def test(feature_matrix, clf):
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
    if sys.argv[1] == 'eval':
        print("EVAL")
        feature_matrix, labels, vectorizer = parse_dataset()
        print("Vectorizer ok")
        clf = eval(feature_matrix, labels)
        print("Model evaluated")
    if sys.argv[1] == 'test':
        print("TEST")
        loaded_model = pickle.load(open(sys.argv[2], 'rb'))
        vectorizer = pickle.load(open(sys.argv[3], 'rb'))
        f_matrix, labels, vectorizer = parse_dataset(TEST_DIR, vectorizer)
        labels = test(f_matrix, loaded_model)
        print(labels)
