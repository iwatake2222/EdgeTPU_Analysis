# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Reshape, Dense, Add, Conv2D, ZeroPadding2D

# Model parameters
MODEL_WIDTH = 2048
MODEL_HEIGHT = 2048
MODEL_CHANNEL = 3

# Create model structure
input0 = Input(shape=(2,2,MODEL_CHANNEL))
pad0 = ZeroPadding2D(padding=((int)(MODEL_WIDTH/2)-1, (int)(MODEL_HEIGHT/2)-1))(input0)
model = Model(inputs=[input0], outputs=[pad0])

# Save model
model.summary()
model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)
model_name = 'model_padding/model_padding_' + str(MODEL_WIDTH) +'x' + str(MODEL_HEIGHT) + 'x' + str(MODEL_CHANNEL)
model.save(model_name + '.h5')

# Convert to quantized tflite
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + '.h5')
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
converter.default_ranges_stats = (0, 6)
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (128., 127.)}  # mean, std_dev
tflite_model = converter.convert()
open(model_name + '.tflite', "wb").write(tflite_model)

'''
tflite_convert ^
  --output_file=model_flatten_2x2x3.tflite ^
  --keras_model_file=model_flatten_2x2x3.h5 ^
  --inference_type=QUANTIZED_UINT8 ^
  --default_ranges_min=0 ^
  --default_ranges_max=6 ^
  --mean_values=128 ^
  --std_dev_values=127

find -name  "*.tflite" -exec 'edgetpu_compiler' '{}' \;
'''
