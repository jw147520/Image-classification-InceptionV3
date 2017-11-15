import pickle
import gzip

dataset_path = ''

def load_batches(verbose=1, samples_per_batch=100):
	# Generator for loading batches of data
	print("Loading dataset file ... ")
	dataset = gzip.open(dataset_path)
	print("Finished loading dataset file.")

	batch_count = 0
	while True:
		try:
			x_train = []
			y_train = []
			x_test = []
			y_test = []

			print('---------- On Batch')

		except:

