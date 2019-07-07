import os
import argparse
import json

from tqdm import tqdm
import cv2

from test_modules import YoloDetector
from test_modules import NNClassifier


read_img = lambda path: cv2.cvtColor(cv2.imread(path),
                                     cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hand and arm detection')
    parser.add_argument('-input_dir', type=str, default='')
    parser.add_argument('-output_dir', type=str, default='')
    args = parser.parse_args()

    input_dir = args.input_dir
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

    lbls_dirs = os.listdir(input_dir)
    for sub_dir in lbls_dirs:
        images_list = os.listdir(os.path.join(input_dir, sub_dir))

        if not os.path.isdir(os.path.join(output_dir, sub_dir)):
            os.makedirs(os.path.join(output_dir, sub_dir))

        for img_name in tqdm(images_list):
            img_path = os.path.join(input_dir,
                                    sub_dir,
                                    img_name)

            img = read_img(img_path)

            detections = detector.predict(img)

            vector_classification = net.predict(img)

            img_json = {'detections': detections,
                        'prob_vector': vector_classification.tolist()}

            output_json = os.path.join(output_dir,
                                       sub_dir,
                                       '{}.json'.format(img_path.split('/')[-1]))

            with open(output_json, 'w') as fid:
                json.dump(img_json, fid)

    
