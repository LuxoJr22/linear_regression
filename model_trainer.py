import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

class model_trainer:
	def __init__(self, file_name):
		self.data = pd.read_csv(file_name)

		self.mileages = self.data.iloc[:, 0]
		self.prices = self.data.iloc[:, 1]
		self.theta0 = 0
		self.theta1 = 0
		self.precision = []

	def normalize(self, data):
		min_value = min(data)
		max_value = max(data)
		return (data - min_value) / (max_value - min_value)

	def train(self, LearningRate = 0.01, epochs = 1000):
		t0 = 0
		t1 = 0

		length = float(len(self.mileages)) 

		norm_mile = self.normalize(self.mileages)
		norm_price = self.normalize(self.prices)


		for i in range(epochs): 
			price_pred = t0 * norm_mile + t1
			D_m = (-2 / length) * sum(norm_mile * (norm_price - price_pred))  
			D_c = (-2 / length) * sum(norm_price - price_pred)
			t0 = t0 - LearningRate * D_m
			t1 = t1 - LearningRate * D_c

			if i % 100 == 0:
				self.precision.append(sum((norm_price - price_pred) ** 2) / length)

		print(self.precision)
		self.theta0 = t0 * (max(self.prices) - min(self.prices)) / (max(self.mileages) - min(self.mileages))
		self.theta1 = (t1 * (max(self.prices) - min(self.prices)) + min(self.prices) - self.theta0 * min(self.mileages))



		price_pred = self.theta0  * self.mileages + self.theta1

		plt.scatter(self.mileages, self.prices)
		plt.plot(self.mileages, price_pred, color='red')
		plt.show()

def main():
	trainer = model_trainer('data.csv')
	trainer.train()
	with open("data.json", "w") as outfile:
		outfile.write(json.dumps({"t0": trainer.theta0 , "t1": trainer.theta1}))
	return

if __name__ == "__main__":
	main()