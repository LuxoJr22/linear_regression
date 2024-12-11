import csv
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (12.0, 8.0)


class model_trainer:
	def __init__(self, datafile):
		self.data = []
		with open(datafile, newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in spamreader:
				self.data += [row]
		for i in range(len(self.data)):
			if i != 0:
				self.data[i] = [float(self.data[i][0]), float(self.data[i][1])]
		
		self.mileages = [item[0] for item in self.data[1:]]
		self.prices = [item[1] for item in self.data[1:]]

		print(self.mileages)
		print(self.prices)

		self.len = len(self.data[1:])
		self.m = 0
		self.c = 0
		self.t0 = 0
		self.t1 = 0

	def normalize(self, data):
		min_value = min(data)
		max_value = max(data)
		return [(x - min_value) / (max_value - min_value) for x in data]


	def train(self, learningRate = 0.1, epoch = 100):
		self.nmile = self.normalize(self.mileages)
		self.nprices = self.normalize(self.prices)
		t0 = 0
		t1 = 0

		print(self.nmile, self.nprices)

		for j in range(epoch):
			der_t0 = 0
			der_t1 = 0

			for i in range(self.len):
				der_t0 += ((t0 + t1 * self.nmile[i]) - self.nprices[i])
				der_t1 += ((t0 + t1 * self.nmile[i]) - self.nprices[i]) * self.nmile[i]
			
			print(der_t0, der_t1)
			
			su = 0
			for l in range(self.len):
				su += ((t0 + t1* self.nmile[l]) - self.nprices[l]) ** 2
			print(su)

			t0 = t0 - learningRate * (der_t0 * (2 / self.len))
			t1 = t1 - learningRate * (der_t1 * (2 / self.len))

		
		print(t0, t1)

		Y_pred = [x * t0 + t1 for x in self.mileages]
		print(Y_pred)
		plt.scatter(self.mileages, self.prices)
		plt.plot([min(self.mileages), max(self.mileages)], [min(Y_pred), max(Y_pred)], color='red')
		plt.show()



def main():
	trainer = model_trainer('data.csv')
	trainer.train()

	return

if __name__ == "__main__":
	main()