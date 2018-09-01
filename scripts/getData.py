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
lstm_days = 30

validation_rate = 0.3

# Advanced parameters
input_macd_days = macd_long_days + macd_signal_days - 1
input_lstm_days = input_macd_days + lstm_days - 1

# Plot settings
plt.style.use('dark_background')

class getter:
	def __init__(self, csv_filename):			
 		self.csv_filename = csv_filename
 		self.parse_flag = False
 		print('Ready to get some data from ' + csv_filename + '!!!')
	
	def get(self):
		if not self.parse_flag:
			# Parse CSV
			self.parseCSV()
			self.parse_flag = True

		return self.in_train, self.out_train, self.in_val, self.out_val

	def getFirstInput(self):
		if not self.parse_flag:
			# Parse CSV
			self.parseCSV()
			self.parse_flag = True

		first_input = self.input_data[:10]
		# return np.reshape(first_input, (1, first_input.shape[0], first_input.shape[1]))
		return first_input

	def getFirstOutput(self):
		if not self.parse_flag:
			# Parse CSV
			self.parseCSV()
			self.parse_flag = True

		first_output = self.output_data[:10]
		# return np.reshape(first_output, (1, first_output.shape[0]))
		return first_output

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

		# Shuffle data
		[self.input_data_random, self.output_data_random] = self.shuffleData(self.input_data, 
			self.output_data)

		# Split data into train and validation 
		[self.in_train, self.out_train, self.in_val, self.out_val] = self.splitData(self.input_data_random, 
			self.output_data_random, validation_rate)

	def buildInputData(self):
		n_available_input = self.close_price.shape[0] - input_lstm_days 
		
		# Get MACD data
		(macd, signal, hist) = self.MACD(self.close_price[:-1], 
			macd_short_days, macd_long_days, macd_signal_days)

		# macd = macd*10
		# signal = signal*10
		# hist = hist*10

		abs_macd_mean = np.mean(np.absolute(macd))
		print("macd absolute mean:\t" + str(abs_macd_mean))
		abs_signal_mean = np.mean(np.absolute(signal))
		print("signal absolute mean:\t" + str(abs_signal_mean))
		abs_hist_mean = np.mean(np.absolute(hist))
		print("hist absolute mean:\t" + str(abs_hist_mean))

		# Create input data (ndarray)
		input_data = []
		for i in range(n_available_input):	
			# this_input = np.concatenate((macd[i:i+lstm_days], signal[i:i+lstm_days]), axis=1)	
			# this_input = np.concatenate((macd[i:i+lstm_days], signal[i:i+lstm_days], hist[i:i+lstm_days]), axis=1)	
			this_input = hist[i:i+lstm_days]
			input_data.append(this_input)
					
		input_data = np.array(input_data)	
		
		return input_data

	def buildOutputData(self):
		n_available_output = self.close_price.shape[0] - input_lstm_days 

		percentage_change_array = np.diff(self.close_price) / np.abs(self.close_price[:-1]) 		
		percentage_change_array = percentage_change_array[-n_available_output:]

		o_array = []
		for o in percentage_change_array:
			if o != 0:
				o_array.append(o/abs(o))
			else:
				o_array.append(0)
		o_array = np.array(o_array)

		output_data = np.reshape(o_array, (o_array.shape[0], 1))
		print(output_data)
		

		# output_data = np.reshape(percentage_change_array, (percentage_change_array.shape[0], 1))
		# print(output_data)

		abs_mean = np.mean(np.absolute(percentage_change_array))
		print("Output absolute mean:\t" + str(abs_mean))

		return output_data

	def shuffleData(self, x, y):
		np.random.seed(None)
		randomList = np.arange(x.shape[0])
		np.random.shuffle(randomList)
		return x[randomList], y[randomList]		

	def splitData(self, x, y, rate):
		x_train = x[int(x.shape[0]*rate):]
		y_train = y[int(x.shape[0]*rate):]
		x_val = x[:int(x.shape[0]*rate)]
		y_val = y[:int(y.shape[0]*rate)]

		return x_train, y_train, x_val, y_val


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


