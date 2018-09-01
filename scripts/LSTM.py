from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, RepeatVector
from keras.layers import LSTM as lstm
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import load_model
from keras import regularizers
import pickle

class LSTM:
	def buildModel(shape, lstm_layers, lstm_units, dense_layers, dense_units):
		print(shape)
		model = Sequential()
		# LSTM layers		
		for l in range(lstm_layers):
			if l == 0:
				this_shape = (shape[1], shape[2])
			else:
				this_shape = (lstm_units, 1)

			if l == lstm_layers - 1:
				this_return = False
			else:
				this_return = True

			model.add(lstm(lstm_units, input_shape=this_shape, return_sequences=this_return))				
		
		# Dense layers
		for l in range(dense_layers):				
			model.add(Dense(dense_units))
			model.add(Dropout(0.3))	
		
		model.add(Dense(1))				
		model.compile(loss="mse", optimizer="adam", metrics=['acc'])
		model.summary()
		return model

	def train(x_train, y_train, x_val, y_val, model, model_name):
		callback_earlystop = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
		callback_tensorboard = TensorBoard(log_dir='../logs/{}'.format(model_name))
		model.fit(x_train, y_train, epochs=20, batch_size=128, 
			validation_data=(x_val, y_val), callbacks=[callback_tensorboard])
		return model

	def saveModel(model, filename):
		model.save(filename)
		print('Saved model to ' + filename)

	def loadModel(filename):		
		return load_model(filename)
		print('Loaded model from ' + filename)

