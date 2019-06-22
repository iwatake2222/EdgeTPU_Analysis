# -*- coding: utf-8 -*-
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D

# Model parameters
MODEL_WIDTH = 128
MODEL_HEIGHT = 128
MODEL_CHANNEL = 3

FILTER_CHANNEL = 32
FILTER_KERNEL_SIZE = 19
FILTER_LAYERS_NUM = 50

# Create model structure
input0 = Input(shape=(MODEL_WIDTH,MODEL_HEIGHT,MODEL_CHANNEL))
conv = input0
for i in range(FILTER_LAYERS_NUM):
	conv = Conv2D(
				filters=FILTER_CHANNEL,
				kernel_size=(FILTER_KERNEL_SIZE,FILTER_KERNEL_SIZE),
				strides=(1,1),
				padding='same',
				activation='relu'
	)(conv)
model = Model(inputs=[input0], outputs=[conv])

# Save model
model.summary()
model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

model.save('model_conv_' + str(FILTER_CHANNEL) +'x' + str(FILTER_KERNEL_SIZE) +'x' + str(FILTER_LAYERS_NUM) + '.h5')

'''
tflite_convert ^
  --output_file=model_conv_32x19x50.tflite ^
  --keras_model_file=model_conv_32x19x50.h5 ^
  --inference_type=QUANTIZED_UINT8 ^
  --default_ranges_min=0 ^
  --default_ranges_max=6 ^
  --mean_values=128 ^
  --std_dev_values=127
'''
