# -*- coding: utf-8 -*-
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2

# Model parameters
MODEL_WIDTH = 4096
MODEL_HEIGHT = 4096
MODEL_CHANNEL = 3


# Create model structure
model_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(MODEL_WIDTH, MODEL_HEIGHT, MODEL_CHANNEL))
model = Model(inputs=model_base.input, outputs=model_base.output)

# Save model
model.summary()
model.compile(
	optimizer='adam',
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

model.save('model_mobilenet_v2_' + str(MODEL_WIDTH) +'x' + str(MODEL_HEIGHT) +'x' + str(MODEL_CHANNEL) + '.h5')

'''
tflite_convert ^
  --output_file=model_mobilenet_v2_4096x4096x3.tflite ^
  --keras_model_file=model_mobilenet_v2_4096x4096x3.h5 ^
  --inference_type=QUANTIZED_UINT8 ^
  --default_ranges_min=0 ^
  --default_ranges_max=6 ^
  --mean_values=128 ^
  --std_dev_values=127
'''