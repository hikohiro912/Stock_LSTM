from LSTM import LSTM
from getData import getter
# parameters
stock_num = '0005'
csv_filename = '../data/' + stock_num + '.HK.csv'
model_filename = '../trained_models/testmodel_' + stock_num + '.h5'


if __name__== "__main__":
	# Load model
	model = LSTM.loadModel(model_filename)
	model.summary()

	# Get latest input
	data_getter = getter(csv_filename)
	input_first = data_getter.getFirstInput()
	output_first = data_getter.getFirstOutput()
	print('first input:\t' + str(input_first))
	print('first output:\t' + str(output_first))

	# Predict
	prediction = model.predict(input_first)
	print('\n************************\n')
	print('Predicted output:\t' + str(prediction))
	err = prediction - output_first	
	percentage_err = err / output_first * 100
	print('Error:\t' + str(err))
	print('Percentage:\t' + str(percentage_err))