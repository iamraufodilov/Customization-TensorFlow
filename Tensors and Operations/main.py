import tensorflow as tf
import numpy as np

x = tf.matmul([[2]], [[2, 3, 4, 5]])
#+>print(x)

my_array = np.ones([3, 7])
#_>print(my_array)

#how to convert ndarray to tensor
my_tensor = tf.multiply(my_array, 111)
#_>print(my_tensor)

#convert tensor to ndarry
my_array1 = np.add(my_tensor, 1)
#_>print(my_array1)

#how to check gpu acceleration
x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
#_>print(tf.config.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
#_>print(x.device.endswith('GPU:0'))


import time

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.config.list_physical_devices("GPU"):
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)


# create dataset
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
print(ds_tensors)

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)
print('our dataset', ds_file)


