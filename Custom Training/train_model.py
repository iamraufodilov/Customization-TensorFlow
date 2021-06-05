# load libraries
import os
import matplotlib.pyplot as plt
import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


# load dataset
data_file_path = 'G:/rauf/STEPBYSTEP/Tutorial/TensorFlow/Customization TensorFlow/Custom Training/iris_training.csv'


# a bit change for dataset
# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

#_>print("Features: {}".format(feature_names))
#_>print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


#Create a tf.data.Dataset
batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    data_file_path,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)


#lets look dataset
features, labels = next(iter(train_dataset))

print(features)
