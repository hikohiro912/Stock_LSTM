from LSTM import LSTM
from getData import getter
# parameters
csv_filename = 'data/0066.HK.csv'

def initialisation():
	# Get data
	data_getter = getter(csv_filename)
	[input_data, output_data] = data_getter.get()

	# Create LSTM
	network = LSTM()

	return input_data, output_data, network

def trainLSTMfromRSI(input_data, output_data, network):
	print('nothing trained!')	


if __name__== "__main__":
	[input_data, output_data, network] = initialisation()
	
	trainLSTMfromRSI(input_data, output_data, network)