# -*- coding: utf-8 -*-
# tensorboard log file 을 사용해 보다 높은 resolution 의 그래프를 그린다.
import tensorflow as tf
import matplotlib.pyplot as plt

# 아래 경로에서 [put event_file_name here] 를 실제 event file 명으로 바꿔야 함.
# ex) path_to_train_events_file = "./log/train/event.out.tfevents ... "
path_to_train_events_file = './log/train/[put event_file_name here]'
path_to_validation_events_file = './log/validation/[put event_file_name here]'

# log file 을 iterate 하며 training step 당 각각의 accuracy/cross-entropy 값을 저장할 lists - y 축 값.
training_accuracy = []
validation_accuracy = []
training_CE = []
validation_CE = []

step = 0  # training step - accuracy
CE_step = 0  # training step - cross entropy
# step = CE_step 이지만 구분하여 사용하였음.
training_x = []  # training-accuracy 그래프를 그리기 위한 x 축 값 list : training step
training_CE_x = []  # training-CrossEntropy 그래프를 그리기 위한 x 축 값 list : training step

# summary_iterator 가 log file 을 돌며 값을 수집.
for e in tf.train.summary_iterator(path_to_train_events_file):
    for v in e.summary.value:
        if v.tag == 'accuracy_1':  # tag 가 'accuracy_1' 인 값을 수집.
            training_accuracy.append(v.simple_value)  # 그래프를 그릴 y 축 list 에 추가.
            training_x.append(step)  # 해당 값이 수집된 training step 을 x 축 list 에 추가.
            # print(v.simple_value)
            step += 1  # training step 증가.
            # <수정> training step 도 tag 로 불러올 수 있을 것.

        if v.tag == 'cross_entropy_1':  # tag 가 'cross_entropy_1' 인 값을 수집
            training_CE.append(v.simple_value)  # 그래프를 그릴 y 축 list 에 추가.
            training_CE_x.append(CE_step)  # 해당 값이 수집된 training step 을 x 축 list 에 추가.
            # print(v.simple_value)
            CE_step += 1  # training step 증가.

step = 0  # training step - accuracy
CE_step = 0  # training step - cross entropy
# step = CE_step 이지만 구분하여 사용하였음.
validation_x = []  # validation-accuracy 그래프를 그리기 위한 x 축 값 list : training step
validation_CE_x = []  # validation-CrossEntropy 그래프를 그리기 위한 x 축 값 list : training step
for e in tf.train.summary_iterator(path_to_validation_events_file):
    for v in e.summary.value:
        if v.tag == 'accuracy_1':
            validation_accuracy.append(v.simple_value)
            validation_x.append(step)
            # print(v.simple_value)
            step += 10
        if v.tag == 'cross_entropy_1':
            validation_CE.append(v.simple_value)
            validation_CE_x.append(CE_step)
            print(v.simple_value)
            CE_step += 10

# <수정> multi plot 으로 바꾸기.
plt.plot(training_CE_x, training_CE)
plt.plot(validation_CE_x, validation_CE)
plt.legend(['Training', 'Validation'], loc='upper right')
plt.ylabel('Cross Entropy')
plt.xlabel('Training step')
plt.savefig('CrossEntropy.png', format='png', dpi=1200)
plt.show()











