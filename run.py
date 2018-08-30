from LSTM import LSTM
from getData import getter
# parameters
csv_filename = 'data/0066.HK.csv'

def initialisation():
	# Get data
	data_getter = getter(csv_filename)
	[in_train, out_train, in_val, out_val] = data_getter.get()

	print('in_train.shape:\t'+str(in_train.shape))
	print('out_train.shape:\t'+str(out_train.shape))
	print('in_val.shape:\t'+str(in_val.shape))
	print('out_val.shape:\t'+str(out_val.shape))

	# Create LSTM
	network = LSTM()

	return in_train, out_train, in_val, out_val, network

def trainLSTMfromRSI(in_train, out_train, in_val, out_val, network):
	print('nothing trained!')	


if __name__== "__main__":
	[in_train, out_train, in_val, out_val, network] = initialisation()
	
	trainLSTMfromRSI(in_train, out_train, in_val, out_val, network)
