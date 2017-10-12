import tensorflow as tf
import matplotlib.pyplot as plt

path_to_train_events_file = './log/train/events.out.tfevents.1501641886.MAZELONESERVER'
path_to_validation_events_file = './log/validation/events.out.tfevents.1501641892.MAZELONESERVER'

training_accuracy = []
validation_accuracy = []
training_CE = []
validation_CE = []

step = 0
CE_step = 0
training_x = []
training_CE_x = []
for e in tf.train.summary_iterator(path_to_train_events_file):
    for v in e.summary.value:
        if v.tag == 'accuracy_1':
            training_accuracy.append(v.simple_value)
            training_x.append(step)
            # print(v.simple_value)
            step += 1
        if v.tag == 'cross_entropy_1':
            training_CE.append(v.simple_value)
            training_CE_x.append(CE_step)
            # print(v.simple_value)
            CE_step += 1

step = 0
CE_step = 0
validation_x = []
validation_CE_x = []
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


plt.plot(training_CE_x, training_CE)
plt.plot(validation_CE_x, validation_CE)
plt.legend(['Training', 'Validation'], loc='upper right')
plt.ylabel('Cross Entropy')
plt.xlabel('Training step')
plt.savefig('CrossEntropy.png', format='png', dpi=1200)
plt.show()











