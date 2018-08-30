import numpy as np 
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Parameters
ema_short_days = 12
ema_long_days = 26
macd_short_days = 12
macd_long_days = 26
macd_signal_days = 9
rsi_days = 14
lstm_days = 7

# Advanced parameters
input_macd_days = macd_long_days + macd_signal_days - 1
input_lstm_days = input_macd_days + lstm_days - 1

# Plot settings
plt.style.use('dark_background')

class getter:
	def __init__(self, csv_filename):			
 		self.csv_filename = csv_filename
 		print('Ready to get some data from ' + csv_filename + '!!!')
	def get(self):
		# Parse CSV
		self.parseCSV()

		return self.input_data, self.output_data

	def parseCSV(self):
		table = pd.read_csv(self.csv_filename)
		table = table.dropna()
		print('Finished reading csv using pandas!')
		
		# Use close price for the prediction 
		self.close_price = np.array(table["Close"])		

		# Build Input Data
		self.input_data = self.buildInputData()

		# Build Output Data
		self.output_data = self.buildOutputData()	


	def buildInputData(self):
		n_available_input = self.close_price.shape[0] - input_lstm_days 
		
		# Get MACD data
		(macd, signal, hist) = self.MACD(self.close_price[:-1], 
			macd_short_days, macd_long_days, macd_signal_days)

		# Create input data (ndarray)
		input_data = []
		for i in range(n_available_input):	
			this_input = np.concatenate((macd[i:i+lstm_days], signal[i:i+lstm_days], hist[i:i+lstm_days]), axis=1)	
			input_data.append(this_input)
					
		input_data = np.array(input_data)	
		
		return input_data

	def buildOutputData(self):
		n_available_output = self.close_price.shape[0] - input_lstm_days 

		percentage_change_array = np.diff(self.close_price) / np.abs(self.close_price[:-1]) 		
		percentage_change_array = percentage_change_array[-n_available_output:]
		output_data = np.reshape(percentage_change_array, (percentage_change_array.shape[0], 1, 1))

		return output_data

	def RSI(self, in_data, days):
		diff_data = np.diff(in_data)
		print('diff_data shape' + str(diff_data.shape))
		# First average gain
		avg_gain = np.sum(np.extract(diff_data[0:days-1]>0, diff_data[0:days-1])) / days
		avg_loss = -np.sum(np.extract(diff_data[0:days-1]<0, diff_data[0:days-1])) / days
		RS = avg_gain / avg_loss
		
		out_data = np.array([100 - 100 / (1 + RS)]) 
		for d in range(days, diff_data.shape[0]):
			this_period = diff_data[d-days+1:d]
			this_avg_gain = np.sum(np.extract(this_period>0, this_period)) / days
			this_avg_loss = -np.sum(np.extract(this_period<0, this_period)) / days
			this_RS = this_avg_gain / this_avg_loss
			this_RSI = 100 - 100 / (1 + this_RS)
			out_data = np.append(out_data, this_RSI)
		
		return out_data

	def SMA(self, in_data, days):
		init_period_sum = np.sum(in_data[0:days-1])
		init_sma = init_period_sum / days

		out_data = np.array([init_sma])		
		for d in range(days,in_data.shape[0]):		
			new_sma = np.sum(in_data[d-days+1:d])/days		
			out_data = np.append(out_data, new_sma)
		
		return out_data

	def EMA(self, in_data, days):
		init_period_sum = np.sum(in_data[0:days-1])
		init_sma = init_period_sum / days
		multiplier = (2/(days+1))

		out_data = np.array([init_sma])
		for d in range(days,in_data.shape[0]):
			new_ema = multiplier*(in_data[d] - out_data[-1]) + out_data[-1]
			out_data = np.append(out_data, new_ema)
	
		return out_data

	def MACD(self, in_data, short_days, long_days, signal_days):
		ema_short = self.EMA(in_data, short_days)
		ema_long = self.EMA(in_data, long_days)

		macd = ema_short[long_days-short_days:] - ema_long

		signal = self.EMA(macd, signal_days)
		macd = macd[signal_days-1:]

		hist = macd - signal	

		macd = np.reshape(macd, (macd.shape[0], 1))
		signal = np.reshape(signal, (signal.shape[0],1))
		hist = np.reshape(hist, (hist.shape[0],1))	

		return macd, signal, hist


