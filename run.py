from LSTM import LSTM
from getData import getter
# parameters
stock_num = '0066'
csv_filename = 'data/' + stock_num + '.HK.csv'
model_filename = 'trained_models/testmodel_' + stock_num + '.h5'

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
	model = LSTM.buildModel(in_train.shape)
	model = LSTM.train(in_train, out_train, in_val, out_val, model)
	LSTM.saveModel(model, model_filename)


if __name__== "__main__":
	[in_train, out_train, in_val, out_val] = getData()
	
	trainLSTM(in_train, out_train, in_val, out_val)
