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

#_>print("This is feature: ",features)
#_>print("this is label: ", labels)
# from result we can see our dictionary data looks have 4 rows which represents feature and 32 collumn with respect batch_size with values of features


# change features to array shape tensor
def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

#_>print(features[:5])
#_>print(labels[:5])
# now our data looks normal as collumns feature names and rows for feature values


# create the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])


predictions = model(features)
#_>predictions[:5] # Here, each example returns a logit for each class.

# To convert these logits to a probability for each class, use the softmax function:
#_>tf.nn.softmax(predictions[:5])

# but our model have not trained sor its prediction has too low accuracy lets look result
#_>print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
#_>print("    Labels: {}".format(labels))


#define the loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels, training=False)
#_>print("Loss test: {}".format(l))


# define gradient
def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


# create optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# lets look single optimization step
loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))
# from here you can see how optimizer much reduce loss in one step


# create training loop
## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
# here we go our model reached 98.3% accuracy in 200 epochs


# Evaluate the model's effectiveness

# load test dataset

test_path = 'G:/rauf/STEPBYSTEP/Tutorial/TensorFlow/Customization TensorFlow/Custom Training/iris_test.csv'

test_dataset = tf.data.experimental.make_csv_dataset(
    test_path,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)


# evalute test dataset
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result())) # here we get 96% accuracy on test dataset


# lets predict unlabelled unseen data with model
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset, training=False)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONCLUSION
'''
in this project we load dataset for classification
then we created tf dataset with right structure
then created loss, gradient and optimization functions
then created model loop
in final we evaluate test dataset and make prediction on custom example
'''