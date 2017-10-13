# -*- coding: utf-8 -*-
# 학습 완료된 classifier 를 테스트한다.
# 학습 완료된 classifier 를 사용해 여러 장의 이미지를 분류해보고 싶을 때도 사용 가능.
# usage command : python test_2.py "image_path"

import tensorflow as tf
import sys
import glob
import csv

# 실행 시 전달받은 image_path 에 접근해 모든 jpg 파일을 불러온다.
image_path = sys.argv[1]
image_path += "/*.jpg"
image_list = glob.glob(image_path)

total = len(image_list)  # image 개수
mal_cnt = 0
ben_cnt = 0

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph_2.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    f = open('final_predictions.csv', 'w', newline='')
    writer = csv.writer(f)

    for image_file in image_list:
        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        # predictions[0][0]: score of benign, prediction[0][1]: score of malignant

        top_k = [0, 0]
        if predictions[0][1] > 0.4:  # Cut-off
            top_k[0] = 1
            mal_cnt += 1
            # writer.writerow([0, 1])
        else:
            top_k[1] = 1
            ben_cnt += 1
            # writer.writerow([0, 0])

        writer.writerow([predictions[0][1], predictions[0][0]])

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
        print('')

print("total: %d" % total)
print("predicted as malignant: %d" % mal_cnt)
print("predicted as benign: %d" % ben_cnt)
