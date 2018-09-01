from LSTM import LSTM
from getData import getter
import time
# parameters
stock_num = '0005'
csv_filename = '../data/' + stock_num + '.HK.csv'
model_filename = '../trained_models/testmodel_' + stock_num + '.h5'
model_name = '{}-stock_model-{}'.format(stock_num, int(time.time()))

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
	model = LSTM.buildModel(in_train.shape, lstm_layers=1, lstm_units=8, 
		dense_layers=0, dense_units=8)
	model = LSTM.train(in_train, out_train, in_val, out_val, model, model_name)
	LSTM.saveModel(model, model_filename)


if __name__== "__main__":
	[in_train, out_train, in_val, out_val] = getData()
	
	trainLSTM(in_train, out_train, in_val, out_val)
