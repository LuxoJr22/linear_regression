import json
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.figsize'] = (10.0, 7.0)

def main():

	data_model = pd.read_csv("data.csv")

	mileages = data_model.iloc[:, 0]
	prices = data_model.iloc[:, 1]

	theta0 = 0
	theta1 = 0
	try:
		file = open('data.json')
		data = json.load(file)
		theta0 = data["t0"]
		theta1 = data["t1"]
	except:
		print("WARNING: You should train the model before predicting")
	print("Write \"stop\" to stop the simulation")

	while (1):
		nb = input("Mileage of the car: ")
		if (nb == "stop"):
			break
		try:
			nb = float(nb)
		except:
			print("Enter a number as input")
			continue
		print(f"A car with {int(nb)}km should cost: {int(theta0 * nb + theta1)}€")

	plt.scatter(mileages, prices)
	price_pred = theta0  * mileages + theta1
	plt.plot(mileages, price_pred, color='red')
	plt.show()

	return

if __name__ == "__main__":
	main()