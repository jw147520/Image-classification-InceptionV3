# -*- coding: utf-8 -*- #
import matplotlib.pyplot as plt

def plot_loss():
	path_train_loss = './log/loss_training.txt'
	path_val_loss = './log/loss_val.txt'

	train_loss = []
	train_x = []
	val_loss = []
	val_x = []

	print("Loading logs...")
	step = 0
	with open(path_train_loss, 'r') as f:
		for v in f.readlines():
			train_loss.append(v)
			train_x.append(step)
			step += 1
			if step > 500:
				break

	step = 0
	with open(path_val_loss, 'r') as f:
		for v in f.readlines():
			val_loss.append(v)
			val_x.append(step)
			step += 1
			if step > 500:
				break
	print("Finished loading logs...")

	print("Drawing graphs")
	plt.plot(train_x, train_loss)
	plt.plot(val_x, val_loss)
	plt.legend(['Training', 'Validation'], loc='upper right')
	plt.ylabel('CrossEntropy')
	plt.xlabel('Training step')
	plt.savefig('./plots/CrossEntropy.png', format='png', dpi=1200)


def plot_acc():
	path_train_acc = './log/acc_training.txt'
	path_val_acc = './log/acc_val.txt'

	train_acc = []
	train_x = []
	val_acc = []
	val_x = []

	print("Loading logs...")
	step = 0
	with open(path_train_acc, 'r') as f:
		for v in f.readlines():
			train_acc.append(v)
			train_x.append(step)
			step += 1
			if step > 10000:
				break

	step = 0
	with open(path_val_acc, 'r') as f:
		for v in f.readlines():
			val_acc.append(v)
			val_x.append(step)
			step += 1
			if step > 10000:
				break
	print("Finished loading logs...")

	print("Drawing graphs")
	plt.plot(train_x, train_acc)
	plt.plot(val_x, val_acc)
	plt.legend(['Training', 'Validation'], loc='lower right')
	plt.ylabel('Accuracy')
	plt.xlabel('Training step')
	plt.savefig('./plots/Accuracy.png', format='png', dpi=1200)


# plot_loss()
plot_acc()