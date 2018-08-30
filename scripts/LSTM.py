from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, RepeatVector
from keras.layers import LSTM as lstm
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import pickle

class LSTM:
	def buildModel(shape):
		print(shape)
		model = Sequential()
		model.add(lstm(16, input_shape=(shape[1], shape[2]), return_sequences=False))	
		model.add(Dropout(0.5))			
		model.add(Dense(1))
		model.compile(loss="mse", optimizer="adam")
		model.summary()
		return model

	def train(x_train, y_train, x_val, y_val, model):
		callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
		model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_data=(x_val, y_val), callbacks=[callback])
		return model

	def saveModel(model, filename):
		model.save(filename)
		print('Saved model to ' + filename)

	def loadModel(filename):		
		return load_model(filename)
		print('Loaded model from ' + filename)

