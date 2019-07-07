import os
import argparse
import json

import cv2

from test_modules import YoloDetector
from test_modules import NNClassifier


read_img = lambda path: cv2.cvtColor(cv2.imread(path),
                                     cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hand and arm detection')
    parser.add_argument('-input_img', type=str, default='')
    parser.add_argument('-output_dir', type=str, default='')
    args = parser.parse_args()

    img_path = args.input_img
    output_dir = args.output_dir

    cfg =     '/src/resources/yolo9k/yolo9000.cfg'
    weights = '/src/resources/yolo9k/yolo9000.weights'
    lbl =     '/src/resources/yolo9k/9k.names'
    detector = YoloDetector(yolo_cfg_path=cfg,
                            yolo_weights_path=weights,
                            labels_file=lbl,
                            threshold=0.1)

    model_path =  '/src/resources/keras/model.json'
    weight_path = '/src/resources/keras/weights.hdf5'
    net = NNClassifier(model_path, weight_path)

    img = read_img(img_path)

    detections = detector.predict(img)

    vector_classification = net.predict(img)

    img_json = {'detections': detections,
                'prob_vector': vector_classification.tolist()}

    output_json = os.path.join(output_dir,
                               '{}.json'.format(img_path.split('/')[-1]))

    with open(output_json, 'w') as fid:
        json.dump(img_json, fid)

    
