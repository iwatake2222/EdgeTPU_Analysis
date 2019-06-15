# -*- coding: utf-8 -*-
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Reshape, Dense, Flatten

# Model parameters
MODEL_WIDTH = 192
MODEL_HEIGHT = 192
MODEL_CHANNEL = 3

# Create model structure
input0 = Input(shape=(MODEL_WIDTH,MODEL_HEIGHT,MODEL_CHANNEL))
reshape0 = Reshape((MODEL_WIDTH * 2, (int)(MODEL_HEIGHT / 2), MODEL_CHANNEL))(input0)
model = Model(inputs=[input0], outputs=[reshape0])

# Save model
model.summary()
model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

model.save('model_reshape_' + str(MODEL_WIDTH) +'x' + str(MODEL_HEIGHT) + 'x' + str(MODEL_CHANNEL) + '.h5')

'''
tflite_convert ^
  --output_file=model_reshape_256x256x3.tflite ^
  --keras_model_file=model_reshape_256x256x3.h5 ^
  --inference_type=QUANTIZED_UINT8 ^
  --default_ranges_min=0 ^
  --default_ranges_max=6 ^
  --mean_values=128 ^
  --std_dev_values=127
'''
