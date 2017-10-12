import tensorflow as tf
import sys
import glob
import csv

# change this as you see fit
image_path = sys.argv[1]
image_path += "/*.jpg"
image_list = glob.glob(image_path)

total = len(image_list)


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

    f = open('benign_cutoff_record.csv', 'w')
    writer = csv.writer(f)

    cutoff = 0.3
    record = []
    while True:
        if cutoff > 0.6:
            break
        mal_cnt = 0
        ben_cnt = 0
        for image_file in image_list:
            # Read in the image_data
            image_data = tf.gfile.FastGFile(image_file, 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            # predictions[0][0]: score of benign, prediction[0][1]: score of malignant

            top_k = [0, 0]

            if predictions[0][1] > cutoff:  # Cut-off
                top_k[0] = 1
                mal_cnt += 1
            else:
                top_k[1] = 1
                ben_cnt += 1

        print("current cutoff: %d, accuracy: %d\n" % (cutoff, (ben_cnt/total)))
        record.append([cutoff, ben_cnt/total])
        writer.writerow([cutoff, ben_cnt/total])
        cutoff += 0.01

    print(record)