from __future__ import division
from collections import defaultdict

class DataFrequency(object):
	
	def __init__(self, labels):
		
		self.labels = labels

	def calculate_label_distribution(self):
		print(len(self.labels))
		d = defaultdict(int)
		for label in self.labels:
			d[label] += 1
		label_frequency_distribution = defaultdict(int)
		for item in d.items():
			label_frequency_distribution[item[0]] += (item[1] / len(self.labels)) * 100
		return label_frequency_distribution
