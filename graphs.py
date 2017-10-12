import matplotlib.pyplot as plt
import csv
import numpy as np
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from scipy import interp
import scikitplot.plotters as skplt


def cutoff_compare():
    cutoff = range(0, 1)
    ben_data = []
    mal_data = []
    ben_datafile = './benign_cutoff_record.csv'
    mal_datafile = './malignant_cutoff_record.csv'

    with open(ben_datafile, 'rt') as f:
        data = csv.reader(f, delimiter=',')
        for d in data:
            ben_data.append(d)

    with open(mal_datafile, 'rt') as f:
        data = csv.reader(f, delimiter=',')
        for d in data:
            mal_data.append(d)

    x = [cutoff for cutoff, acc in ben_data]
    y_ben = [acc for cutoff, acc in ben_data]
    y_mal = [acc for cutoff, acc in mal_data]
    print(x)
    print(y_ben)
    print(y_mal)

    p1 = plt.plot(x, y_ben, 'r^--', label='benign')
    p2 = plt.plot(x, y_mal, 'bs-', label='malignant')

    plt.legend()
    plt.title('Cutoff - Accuracy')
    plt.xlabel('CutOff'), plt.ylabel('Accuracy')
    plt.show()


def draw_roc():
    y_label = []
    y_prediction = []
    with open('predictions.csv', 'rt') as f:
        data = csv.reader(f, delimiter=',')
        for d in data:
            if d[0] == '0':
                y_label.append('benign')
            else:
                y_label.append('malignant')
            # y_label.append(int(d[0]))

    with open('final_predictions.csv', 'rt') as f:
        data = csv.reader(f, delimiter=',')
        for d in data:
            y_prediction.append([float(d[1]), float(d[0])])

    print(y_label)
    print(y_prediction)

    skplt.plot_roc_curve(y_label, y_prediction)
    plt.show()


if __name__ == '__main__':
    # cutoff_compare()
    draw_roc()
