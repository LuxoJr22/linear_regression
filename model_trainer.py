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
		self.pr_i = []

	def normalize(self, data):
		min_value = min(data)
		max_value = max(data)
		return (data - min_value) / (max_value - min_value)

	def stats(self):
		price_pred = self.theta0  * self.mileages + self.theta1

		plt.scatter(self.mileages, self.prices)
		plt.plot(self.mileages, price_pred, color='red')
		plt.title("Prices Predictions")
		plt.xlabel("Price")
		plt.ylabel("Mileage")
		plt.show()


		plt.plot(self.pr_i, self.precision, color='red')
		plt.title("Mean squarred error")
		plt.xlabel("Epoch")
		plt.ylabel("Error (normalised)")
		plt.show()

		v = ((self.prices - price_pred)**2).sum()
		m = ((self.prices - self.prices.mean())**2).sum()
		print(f"The precision of the model is: {round((1 - v/m) * 100, 2)}%")



	def train(self, LearningRate = 0.1, epochs = 1000):
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

			self.precision.append(sum((norm_price - price_pred) ** 2) / length)
			self.pr_i.append(i)

		self.theta0 = t0 * (max(self.prices) - min(self.prices)) / (max(self.mileages) - min(self.mileages))
		self.theta1 = (t1 * (max(self.prices) - min(self.prices)) + min(self.prices) - self.theta0 * min(self.mileages))


def main():
	trainer = model_trainer('data.csv')
	trainer.train()
	with open("data.json", "w") as outfile:
		outfile.write(json.dumps({"t0": trainer.theta0 , "t1": trainer.theta1}))
	trainer.stats()
	return

if __name__ == "__main__":
	main()