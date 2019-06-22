import time
from PIL import Image, ImageDraw, ImageFont
import numpy
from edgetpu.basic.basic_engine import BasicEngine

MODEL_NAME = "model_mobilenet/mobilenet_v1_1.0_224_quant.tflite"


### Load model and prepare TPU engine
engine = BasicEngine(MODEL_NAME)
width = engine.get_input_tensor_shape()[1]
height = engine.get_input_tensor_shape()[2]

### prepara input tensor
img = Image.new('RGB', (width, height), (128, 128, 128))
# img = Image.new('RGB', (width, height), (127, 128, 129))

# imarray = numpy.random.rand(width,height,3) * 255
# img = Image.fromarray(imarray.astype('uint8')).convert('RGB')

draw = ImageDraw.Draw(img)
input_tensor = numpy.asarray(img).flatten()

### Run inference
start = time.time()
num_measurement = 500
for i in range(num_measurement):
	_, raw_result = engine.RunInference(input_tensor)
elapsed_time = time.time() - start
print ("elapsed_time: {0} ".format(1000 * elapsed_time / num_measurement) + "[msec]")

# for wireshark analysis
# start = time.time()
# num_measurement = 5
# for i in range(num_measurement):
# 	_, raw_result = engine.RunInference(input_tensor)
# 	time.sleep(2)
# elapsed_time = time.time() - start
# print ("elapsed_time: {0} ".format(1000 * elapsed_time / num_measurement) + "[msec]")

