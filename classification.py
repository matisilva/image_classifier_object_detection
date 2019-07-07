import os
import sys
import pickle
import csv
import json
import numpy as np
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
from sklearn.model_selection import GridSearchCV


TRAIN_DIR = "./train_annotations_json_2"
TEST_DIR = "./test_annotations_json"
LABELS_FILE = "./train.csv"


class OwnVotingClassifier(VotingClassifier):
    def __init__(self, estimators, voting='hard', weights=None, n_jobs=None,
                 flatten_transform=True):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform

    # def _collect_probas(self, X):
    #     _probs = np.asarray([clf.predict_proba(X) for clf in self.estimators_])
    #     for vote in self.external_votes:
    #         np.append(_probs, vote)
    #     return _probs

    def _predict_proba(self, X):
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        return np.average(self._collect_probas(X), axis=0,
                          weights=self._weights_not_none)
        # tunned_results = []
        # for index, x in enumerate(X):
        #     result = np.asarray(self.external_votes[index])[0]
        #     if np.any(x):
        #         result = np.average(self._collect_probas([x]), axis=0,
        #                             weights=self._weights_not_none)[0]
        #     tunned_results.append(result)
        # print(np.asarray(tunned_results).shape)
        # return np.asarray(tunned_results)


def parse_annotated_labels():
    annotated_labels = dict()
    print("Getting labels from annotations")
    with open(LABELS_FILE) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in tqdm(csvReader):
            annotated_labels[row[1]] = row[2]
    return annotated_labels


def parse_dataset_json(annotations_dir=TRAIN_DIR, vectorizer=None):
    mode = 'test'
    if vectorizer is None:
        mode = 'train'
    categories = os.listdir(annotations_dir)
    print("Making training matrix")
    featurized_imgs = []
    matrix = []
    labels = []
    external_votes = []
    if mode == 'train':
        vectorizer = DictVectorizer(sparse=False)
        annotated_labels = parse_annotated_labels()
    for category in tqdm(categories):
        if category.startswith("."):
            continue
        for file_id in os.listdir(os.path.join(annotations_dir, category))[:100]:
            featurized_img = defaultdict(int)
            with open(os.path.join(annotations_dir, category, file_id)) as f:
                annotations = json.load(f)
            for tag in annotations['detections']:
                featurized_img[tag[1]] += 1
            featurized_imgs.append(featurized_img)
            external_votes.append(annotations['prob_vector'])
            if mode == 'train':
                class_img = annotated_labels[file_id.split(".jpg")[0]]
                labels.append(class_img)
    if mode == 'train':
        matrix = vectorizer.fit_transform(featurized_imgs)
    else:
        matrix = vectorizer.transform(featurized_imgs)
    return matrix, labels, vectorizer, external_votes


def make_model(model_type=None):
    clf1 = LogisticRegression(solver='lbfgs',
                              multi_class='multinomial',
                              random_state=1,
                              max_iter=700)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    clf4 = svm.SVC(gamma='scale', probability=True)
    # estimators = [('lr', clf1), ('rf', clf2), ('gnb', clf3), ('svm', clf4)]
    estimators = [('rf', clf2), ('svm', clf4)]

    # Use the key for the classifier followed by __ and the attribute
    params = {'rf__n_estimators': [50, 100, 500],
              'svm__C': [0.1, 0.5, 0.8, 1, 3, 4],
              'svm__kernel': ['rbf', 'poly', 'linear'],
              'svm__gamma': [0.66, 0.77, 0.88, 0.99, 1.5],
              'svm__class_weight': ['balanced', None]}

    eclf = OwnVotingClassifier(estimators=estimators, voting='soft',
                               flatten_transform=True)
    if model_type == 'grid':
        return GridSearchCV(estimator=eclf, param_grid=params, cv=2)

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


def eval(feature_matrix, labels, model_type=None, save=True,
         external_votes=None):
    clf = make_model(model_type)
    print(clf.__class__.__name__)
    if not clf:
        raise Exception("No clf setted")
    print('Model evaluation')
    ext_votes = external_votes
    tr_m, te_m, tr_labels, te_labels, tr_ext_v, te_ext_v = train_test_split(
        feature_matrix, labels, ext_votes, test_size=0.33, random_state=42)
    clf.external_votes = tr_ext_v  #  This only works if is OwnVotingClassifier
    clf.fit(tr_m, tr_labels)
    if model_type == 'grid':
        print(clf.best_params_)
    clf.external_votes = te_ext_v  #  This only works if is OwnVotingClassifier
    pred_labels = clf.predict(te_m)
    print(classification_report(te_labels, pred_labels))
    print(balanced_accuracy_score(te_labels, pred_labels))
    return clf


def test(feature_matrix, clf):
    labels = clf.predict(feature_matrix)
    return labels


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("BAD ARGS")
    if sys.argv[1] == 'train':
        print("TRAIN")
        feature_matrix, labels, vectorizer = parse_dataset_json()
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
        feature_matrix, labels, vectorizer, ext_votes = parse_dataset_json()
        print("Vectorizer ok")
        clf = eval(feature_matrix, labels, model_type='grid',
                   external_votes=ext_votes)
        print("Model evaluated")
    if sys.argv[1] == 'test':
        print("TEST")
        loaded_model = pickle.load(open(sys.argv[2], 'rb'))
        vectorizer = pickle.load(open(sys.argv[3], 'rb'))
        f_matrix, labels, vectorizer = parse_dataset_json(TEST_DIR, vectorizer)
        labels = test(f_matrix, loaded_model)
        print(labels)
