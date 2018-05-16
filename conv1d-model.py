import argparse
import pandas as pd
import numpy as np
np.random.seed(2500)
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
from laplotter import LossAccPlotter
from data_frequency import DataFrequency
import plotly.plotly as py
import plotly.graph_objs as go


def load_dataset(data_file):
	
	if (data_file):
		df = pd.read_csv(data_file, header=None)
		data_array = df.values
		x, y = [],[]
		for data in data_array:
			x.append(data[1:])
			y.append(data[0])
		x = np.array(x)
		y = np.array(y)
	else:
		raise IOError('Non-empty filename expected.')
	return x,y

def create_train_test(x, y):
	
	x_train, x_test, y_train, y_test = \
		train_test_split(x, y, test_size=0.30, random_state=42)
	
	return x_train, x_test, y_train, y_test

def transform_data(x_train, x_test, y_train, y_test):

	# reshape attributes to spatial dimensions
	x_train = np.expand_dims(x_train, axis=2)
	x_test = np.expand_dims(x_test, axis=2)

	# one-hot-encode output labels (protocol names)
	encoder = LabelEncoder()
	encoder.fit(y_train)
	encoded_y_train = encoder.transform(y_train)
	y_train = np_utils.to_categorical(encoded_y_train)

	encoder.fit(y_test)
	class_labels = encoder.classes_ 							# the name of the class labels encoded
	nb_classes = len(class_labels)								# the number of different labels being trained
	encoded_y_test = encoder.transform(y_test)
	y_test = np_utils.to_categorical(encoded_y_test)
		
	return x_train, x_test, y_train, y_test, class_labels, nb_classes

def build_model(nb_classes, optimizer, show_summary=False):
	"""
	Build and return a CNN model
	:param show_summary: boole flag to show built model summary
	:return: tensorflow keras model
	"""

	model = Sequential()
	model.add(Conv1D(filters=2, kernel_size=5,
			activation='relu', input_shape=(1024,1)))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(16, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(nb_classes, activation='softmax'))

	if optimizer == 'SGD':
		gd_optimizer = optimizers.SGD(lr=0.001)

	elif optimizer == 'Adam':
		gd_optimizer = optimizers.Adam(lr=0.001)

	elif optimizer == 'RMSprop':
		gd_optimizer = optimizers.RMSprop(lr=0.001)

	elif optimizer == 'SGD-Momentum':
		gd_optimizer = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False)

	elif optimizer == 'SGD-Nesterov':
		gd_optimizer = optimizers.SGD(lr=0.001, momentum=0.8, nesterov=True)
	
	else:
		raise IOError('Uknown optimizer specified.')

	model.compile(optimizer=gd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	
	if show_summary:
		model.summary()
	return model

def run_model(mode, model, optimizer, x_features, y_labels, *class_labels):
	"""	
	:param mode: one of the modes from {TRAIN, EVAL, PRED}
	:param model: keras 1D convolutional model
	:param x_features: input data for the model (attributes/features)
	:param y_labels: labels associated with input data
	"""

	if mode == 'PRED':
		preds = model.predict(x_features, batch_size=32, verbose=0)
		cm_file = plot_confusion_matrix(y_labels, preds, class_labels, optimizer)
		plot_class_performance_metric(y_labels, preds, class_labels, optimizer)
		msg = "Saved confusion matrix to {}".format(cm_file)
		
	elif mode == 'EVAL':
		loss, accuracy = model.evaluate(x_features, y_labels, verbose=0)
		msg = "\nModel evaluation finished\nLoss: {}\tAccuracy: {}".format(loss, accuracy)

	else:
		saved_model_file = 'models/trained_model_{}.h5'.format('conv1d-' + optimizer)		
		weights = 'weights/weight_model{}.hdf5'.format('conv1d-' + optimizer)

		#try:
		#	model.load_weights(weights)		
		#except:
		#	raise IOError('Specified weight file does not exist.')
		
		#save model at checkpoints when loss function improved
		checkpoint = ModelCheckpoint(saved_model_file, monitor='val_loss', 
											save_best_only=True, verbose=1)		
		fit_history = model.fit(x_features, y_labels, epochs=200, batch_size=32, 
				 						validation_split=0.10, callbacks=[checkpoint])
		
		plot_loss_acc_history(fit_history.history, optimizer)		

		model.save_weights(weights)
		msg = "Model training finished"

	print(msg)

def plot_confusion_matrix(y_labels, preds, class_labels, optimizer_name):

	y_true_labels = [np.argmax(t) for t in y_labels]
	y_preds_labels = [np.argmax(t) for t in preds]

	cm = confusion_matrix(y_true_labels, y_preds_labels)

	df_cm = pd.DataFrame(cm)
	plt.figure(figsize=(20,15))
	plt.xlabel('Predicted')
	plt.ylabel('True')
	fig = sn.heatmap(df_cm, cmap='coolwarm', xticklabels=class_labels[0],
					yticklabels=class_labels[0], linewidths=.5, annot=True, fmt="d")
	pdf_filename = 'confusion_matrix/' + optimizer_name + '-' + 'confusion_matrix.pdf'
	fig.get_figure().savefig(pdf_filename, dpi=400)
	return pdf_filename

def plot_class_performance_metric(y_labels, preds, class_labels, optimizer_name):

	y_true_labels = [np.argmax(t) for t in y_labels]
	y_preds_labels = [np.argmax(t) for t in preds]

	class_metric_report = classification_report(y_true_labels, y_preds_labels, \
										target_names=class_labels[0], digits=4)
	print(class_metric_report)

	class_precision = precision_score(y_true_labels, y_preds_labels, average=None)
	class_recall = recall_score(y_true_labels, y_preds_labels, average=None)
	class_f1_score = f1_score(y_true_labels, y_preds_labels, average=None)
	
	metrics = {'labels': class_labels[0][7:], 'Precision': class_precision[7:], \
							'Recall': class_recall[7:], 'F1': class_f1_score[7:] }
	
	df_metrics = pd.DataFrame(metrics, index=metrics['labels'])
	df_metrics.plot.bar(figsize=(26,18), fontsize=16, grid=True, rot=25)
	plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
	plt.xlabel('Protocol Label', fontsize=20)
	plt.ylabel('Performance Metric [0-1]', fontsize=20)
	plt.yticks(np.arange(0.0, 1.05, 0.05), fontsize=20)
	plt.savefig('metrics/' + optimizer + '-' + 'performance_metric.png', dpi=400),
		
def plot_label_frequecy(label_frequency_distribution):

 	label_frequency = sorted(label_frequency_distribution.items())
	keys = sorted(label_frequency_distribution.keys())
	df_label_freq = pd.DataFrame(label_frequency[7:], index=keys[7:], \
							columns=['protocol','frequency'])
	df_label_freq.plot.bar(figsize=(20,15), grid=True, fontsize=16, rot=30)
	plt.xlabel('Protocol Label', fontsize=20)
	plt.ylabel('Frequency (%)', fontsize=20)
	plt.yticks(np.arange(0.0, 10.05, 0.5), fontsize=20)
	plt.savefig('frequency_data/frequency_plot.png', dpi=600)
	
def plot_loss_acc_history(fit_history, optimizer_name):
	
	plotter = LossAccPlotter(title = optimizer + ': Loss and Accuracy Performance',
							save_to_filepath='loss_acc_plots/' + optimizer + '.png',
							show_regressions=True,
							show_averages=False,
							show_loss_plot=True,
							show_acc_plot=True,
							show_plot_window=True,
							x_label="Epoch")

	num_epochs = len(fit_history['acc'])

	for epoch in range(num_epochs):
		
		acc_train = fit_history['acc'][epoch]
		loss_train = fit_history['loss'][epoch]
		
		acc_val = fit_history['val_acc'][epoch]
		loss_val = fit_history['val_loss'][epoch]
		
		plotter.add_values(epoch, loss_train=loss_train, acc_train=acc_train,
							loss_val=loss_val, acc_val=acc_val, redraw=False)

	plotter.redraw()
	plotter.block()	

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--run-mode', required=True, type=str, choices=['TRAIN', 'EVAL', 'PRED'],
						help='TRAIN : train model, EVAL : evaluate model, PRED : make prediction with model')
	parser.add_argument('--data-csv', required=True, type=str, help='Raw data CSV file for input')
	parser.add_argument('--optimizer', required=True, type=str, choices=['SGD', 'Adam', 'RMSprop', \
						'SGD-Momentum', 'SGD-Nesterov'], help='Name of optimizer to train the deep learning model')

	args = parser.parse_args()

	# Load data
	x, y = load_dataset(args.data_csv)

	# Calculate protocol label frequency distribution
	data_freq = DataFrequency(y)
	label_frequency_distribution = data_freq.calculate_label_distribution()
	# Plot bar chart
	plot_label_frequecy(label_frequency_distribution)

	# Create train and test set
	x_train, x_test, y_train, y_test = create_train_test(x, y)	

	# Transform data
	x_train, x_test, y_train, y_test, class_labels, nb_classes = \
			transform_data(x_train, x_test, y_train, y_test) 

	# Get the name of the optimizer 
	optimizer = args.optimizer
	
	if args.run_mode ==	'TRAIN':
		# Create model
		cnn_model = build_model(nb_classes, optimizer)
		# Run model for TRAINING
		run_model(args.run_mode, cnn_model, optimizer, x_train, y_train)
	else:
		try:
			# Load mode
			cnn_model = load_model('models/trained_model_{}.h5'.format('conv1d-' + optimizer))	
		except:
			raise IOError('Model for the specified optimizer not created yet. Train the model first.')
		# Evaluate or make prediction		
		run_model(args.run_mode, cnn_model, optimizer, x_test, y_test, class_labels)
