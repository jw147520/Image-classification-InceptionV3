# -*- coding: utf-8 -*-
# 학습 완료된 classifier 를 테스트한다.
# 학습 완료된 classifier 를 사용해 여러 장의 이미지를 분류해보고 싶을 때도 사용 가능.
# usage command : python test.py [image_path]

import tensorflow as tf
import os
import sys
import glob
import csv

# 실행 시 전달받은 image_path 에 접근해 모든 jpg 파일을 불러온다.
image_path = []
for dir_name in os.listdir(sys.argv[1]):  # 원본 이미지 경로에 있는 모든 하위 디렉토리(즉 모든 label)
    image_path.append(sys.argv[1] + "/" + dir_name + "/*.jpg")

image_list = [] # 해당 경로에 존재하는 이미지 파일들의 리스트
for label_path in image_path:
    temp_list = glob.glob(label_path)
    image_list += temp_list

total = len(image_list)  # image 개수
print("Total number of test images : %d" % total)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

if len(image_path) != len(label_lines):
    print("Number of classes are not matched!!")
    print("Number of classes in test data directory: %d" % len(image_path))
    print("Number of classes in retrained_labels.txt: %d" % len(label_lines))
    quit()

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:  # 학습 완료된 모델을 불러온다.
    graph_def = tf.GraphDef()  # 기본 그래프 구조
    graph_def.ParseFromString(f.read())  # 학습 완료된 모델에서 값들을 불러와 그래프를 구성한다.
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # 이후 분석을 위한 csv 파일 생성
    f = open('final_predictions.csv', 'w', newline='')
    writer = csv.writer(f)

    print("\nClassifying test images ... ")
    for image_file in image_list:
        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # predictions[0][i] 에는 prediction 의 결과 class i 일 score 가 저장되어 있다.
        # 즉, predictions[0] 은 score list.
        # 아래와 같이 score 에 따라 내림차순으로 정렬한다.
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        print("File: " + image_file)
        print("----- Classification scores for each classes -----")
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print("%s : score = %.5f" % (human_string, score))
        print('')
