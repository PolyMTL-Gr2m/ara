from keras2c import k2c
#from tensorflow import keras
import os
from huggingface_hub import from_pretrained_keras
import keras
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# Import python libraries required in this example:
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# Use numpy arrays to store inputs (x) and outputs (y):
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Define the network model and its arguments.
# Set the number of neurons/nodes for each layer:
model = Sequential()
model.add(Dense(2, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))

# Compile the model and calculate its accuracy:
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

# Print a summary of the Keras model:
model.summary()

#model = from_pretrained_keras()


def random_tensor(shape):
  return np.random.rand(*shape)

def float_to_nbit_uint(x, n=8):
  # Convert the tensor to an n-bit unsigned int tensor
  x_int = [(int(i * (2 ** n - 1)) & (2 ** n - 1)) for i in x]
  return x_int
try:
  os.remove("resnet20.c")
except:
  print("resnet20.c not found")
try:
  os.remove("resnet20.h")
except:
  print("resnet20.h not found")
try:
  os.remove("resnet20_test_suite.c")
except:
  print("resnet20_test_suite.c not found")
x  = [0.11,0.22,0.333,0.444,0.424,0.283746]

# Convert the tensor to an 8-bit unsigned int tensor
x_int = float_to_nbit_uint(x)

# Print the tensor
print(x_int)  # [1 2 3]

print("done")
model = keras.models.load_model("convnet.h5")
#model = ResNet50(weights='imagenet')
model.summary()

#flops = tf.profiler.profile(graph,\
             #options=tf.profiler.ProfileOptionBuilder.float_operation())
#print('FLOP = ', flops.total_float_ops)

function_name = 'resnet20'
#function_name = 'simplenet'
# allowed datatype for now are 'float ', 'int ', and 'int8_t ', and 'bool '
k2c(model, function_name, malloc=False, num_tests=1, verbose=True, datatype='int8_t ')
