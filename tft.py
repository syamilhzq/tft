# Library
import pandas as pd 
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

#temporal fusion layer
class TemporalFusion(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(TemporalFusion, self).__init__(**kwargs)
        self.units = units
        
    def build(self, batch_input_shape):
        self.w1 = self.add_weight(shape=(batch_input_shape[0][-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.w2 = self.add_weight(shape=(batch_input_shape[1][-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        super(TemporalFusion, self).build(batch_input_shape)
        
    def call(self, inputs):
        x1, x2 = inputs
        x1 = tf.matmul(x1, self.w1)
        x2 = tf.matmul(x2, self.w2)
        x = x1 + x2 + self.b
        return x
    
    def compute_output_shape(self, batch_input_shape):
        return (None, self.units)

#load the data
dataframe = pd.read_csv('300data.csv', usecols=[1], engine='python')
dataset = dataframe.values

#normalize the data to be in the range of 0-1
normalize_data = MinMaxScaler(feature_range=(0, 1))
dataset = normalize_data.fit_transform(dataset)

#split the data into training and test sets
training_data = int(len(dataset) * 0.3)
x_train, x_test = dataset[0:training_data, 0], dataset[training_data:len(dataset), 0]
y_train, y_test = dataset[0:training_data, 0], dataset[training_data:len(dataset), 0]
print(len(x_train), len(x_test))

#TFT model
inputs1 = keras.layers.Input(shape=(1,))
inputs2 = keras.layers.Input(shape=(1,))
fusion = TemporalFusion(units=64)([inputs1, inputs2])
predictions = keras.layers.Dense(1)(fusion)
model = keras.models.Model(inputs=[inputs1, inputs2], outputs=predictions)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit([x_train, x_train], y_train, epochs=50, verbose=2)

# Make predictions on the test set
predictions = model.predict([x_test, x_test])

#inverse the data to original value
predictions = normalize_data.inverse_transform(predictions)

#shift test predictions for plotting
prediction_plot = np.empty_like(dataset)
prediction_plot[:, :] = np.nan
prediction_plot[len(x_train):len(dataset), :] = predictions

# Plot the predictions and actual values
plt.plot(normalize_data.inverse_transform(dataset))
plt.plot(prediction_plot)
plt.show()