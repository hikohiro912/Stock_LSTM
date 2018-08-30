import numpy as np 
import pandas as pd
import csv
import matplotlib.pyplot as plt

class getter:
	def __init__(self, csv_filename):			
 		self.csv_filename = csv_filename
 		print('Ready to get some data from ' + csv_filename + '!!!')
	def get(self):
		# Parse CSV
		self.parseCSV()

		input_data = [1,2,3]
		output_data = [4,5,6]
		return input_data, output_data

	def parseCSV(self):
		table = pd.read_csv(self.csv_filename)
		print('Finished reading csv using pandas')
		nan_rows = table[table['Close'].isnull()]
		print(nan_rows)
		self.close_price = np.array(table["Close"])
		
		sma = self.SMA(self.close_price, 10)

		###### Plot ######
		# plt.plot(self.close_price)
		# plt.show()
		

	def RSI(self, in_data):


		return out_data

	def SMA(self, in_data, days):
		init_period_sum = np.sum(in_data[0:days-1])
		init_sma = init_period_sum / days

		out_data = np.array([init_sma])		
		for d in range(days,in_data.shape[0]):				
			new_sma = out_data[-1] + in_data[d]/days - in_data[d-1]/days			
			out_data = np.append(out_data, new_sma)

		t = np.isnan(out_data)
		plt.plot(t)
		plt.show()
		return out_data

	def EMA(self, in_data, days):
		

		return out_data

	def MACD(self, in_data):

		return out_data


