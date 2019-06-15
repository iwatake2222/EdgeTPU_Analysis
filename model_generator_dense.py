# -*- coding: utf-8 -*-
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Flatten, Dense

# Model parameters
MODEL_WIDTH = 128
MODEL_HEIGHT = 128
MODEL_CHANNEL = 3

DENSE_UNITS_NUM = 500
DENSE_LAYERS_NUM = 10

# Create model structure
input0 = Input(shape=(MODEL_WIDTH,MODEL_HEIGHT,MODEL_CHANNEL))
flatten0 = Flatten()(input0)
flatten1 = Dense(units=2, activation='relu')(flatten0)    # decrease input data size (128x128x200 -> 128*128*2)
for i in range(DENSE_LAYERS_NUM):
	flatten1 = Dense(units=DENSE_UNITS_NUM, activation='relu')(flatten1)
model = Model(inputs=[input0], outputs=[flatten1])

# Save model
model.summary()
model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

model.save('model_dense_' + str(DENSE_UNITS_NUM) +'x' + str(DENSE_LAYERS_NUM) + '.h5')

'''
tflite_convert ^
  --output_file=model_dense_500x10.tflite ^
  --keras_model_file=model_dense_500x10.h5 ^
  --inference_type=QUANTIZED_UINT8 ^
  --default_ranges_min=0 ^
  --default_ranges_max=6 ^
  --mean_values=128 ^
  --std_dev_values=127
'''
