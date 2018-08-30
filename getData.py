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
lstm_days = 2

plt.style.use('dark_background')

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
		table = table.dropna()
		print('Finished reading csv using pandas!')
		
		self.close_price = np.array(table["Close"])
		print('close price shape:\t'+str(self.close_price.shape[0]))	
		
		# ema_short = self.EMA(self.close_price, ema_short_days)
		# ema_long = self.EMA(self.close_price, ema_long_days)

		(macd, signal, hist) = self.MACD(self.close_price[:-1], 
			macd_short_days, macd_long_days, macd_signal_days)
		print('macd shape:\t'+str(macd.shape[0]))

		input_macd_days = macd_long_days + macd_signal_days - 1
		input_lstm_days = input_macd_days + lstm_days - 1
		print('input_macd_days:\t'+str(input_macd_days))
		print('input_lstm_days:\t'+str(input_lstm_days))

		n_available_input = self.close_price.shape[0] - input_lstm_days 
		percentage_change_array = np.diff(self.close_price) / np.abs(self.close_price[:-1]) 
		print(percentage_change_array)
		output_data = percentage_change_array[-n_available_input:]

		print('output_data:\t' + str(output_data.shape[0]))
		print('n_available input:\t' + str(n_available_input))

		# Build Input Data
		input_data = self.buildInputData(macd, lstm_days)




	def buildInputData(self, input_data, past_days):
		result = []
		for i in range(0, input_data.shape[0]-past_days):
			
			result = np.append(result, input_data[i:i+past_days], axis=0)
			
		print(result.shape)
		return result


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

		return macd, signal, hist


