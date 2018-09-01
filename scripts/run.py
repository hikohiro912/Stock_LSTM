from LSTM import LSTM
from getData import getter
import time
# parameters
stock_num = '0005'
csv_filename = '../data/' + stock_num + '.HK.csv'
model_filename = '../trained_models/testmodel_' + stock_num + '.h5'

# model settings
lstm_l = 1
lstm_u = 8
dense_l = 1
dense_u = 32
model_name = '{}-LSTM-{},{}-Dense{},{}-{}'.format(stock_num, lstm_l, lstm_u,
	dense_l, dense_u, int(time.time()))

def getData():
	# Get data
	data_getter = getter(csv_filename)
	[in_train, out_train, in_val, out_val] = data_getter.get()

	print('in_train.shape:\t'+str(in_train.shape))
	print('out_train.shape:\t'+str(out_train.shape))
	print('in_val.shape:\t'+str(in_val.shape))
	print('out_val.shape:\t'+str(out_val.shape))

	return in_train, out_train, in_val, out_val

def trainLSTM(in_train, out_train, in_val, out_val):
	# Create LSTM model
	model = LSTM.buildModel(in_train.shape, lstm_layers=lstm_l, lstm_units=lstm_u, 
		dense_layers=dense_l, dense_units=dense_u)
	model = LSTM.train(in_train, out_train, in_val, out_val, model, model_name)
	LSTM.saveModel(model, model_filename)


if __name__== "__main__":
	[in_train, out_train, in_val, out_val] = getData()
	
	trainLSTM(in_train, out_train, in_val, out_val)
