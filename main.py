import csv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14.0, 9.0)


class model_trainer:
	def __init__(self, datafile):
		self.data = []
		with open(datafile, newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in spamreader:
				self.data += [row]
		for i in range(len(self.data)):
			if i != 0:
				self.data[i] = [float(self.data[i][0]) / 1000, float(self.data[i][1]) / 1000]
		
		self.mileages = [item[0] for item in self.data[1:]]
		self.prices = [item[1] for item in self.data[1:]]

		self.len = len(self.data[1:])
		self.m = 0
		self.c = 0
	
	def train_loop(self, learningRate):
		
		price_pred = [x * self.m + self.c for x in self.mileages]

		D_m = (-2/self.len) * sum(self.mileages * np.subtract(self.prices, price_pred ))
		D_c = (-2/self.len) * sum(np.subtract(self.prices, price_pred))


		self.m = self.m - learningRate * D_m
		self.c = self.c - learningRate * D_c




	def train(self, learningRate = 0.001, epoch = 10):
		for i in range(epoch):
			self.train_loop(learningRate)

		Y_pred = [x * self.m + self.c for x in self.mileages]
		plt.scatter(self.mileages, self.prices)
		plt.plot([min(self.mileages), max(self.mileages)], [min(Y_pred), max(Y_pred)], color='red')
		plt.show()



def main():
	trainer = model_trainer('data.csv')
	trainer.train()

	return

if __name__ == "__main__":
	main()