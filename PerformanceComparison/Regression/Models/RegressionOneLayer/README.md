### Model Description:
The following code snippet illustrates the structure of the model:

```python
input_dimension = x_train.shape[1]
output_dimension = y_train.shape[1]

layer_complexity_growth = 1.5
l1_scaling = 0.001
hidden_layer_1 = round(input_dimension * layer_complexity_growth)
output_layer = 1

# Placeholder for batch of inputs:
x = tf.placeholder(tf.float32, [None, input_dimension])

# Layer 1 Variables:
W1 = tf.Variable(tf.truncated_normal([input_dimension, hidden_layer_1], stddev=0.15))
b1 = tf.Variable(tf.zeros([hidden_layer_1]))
y1 = l1_scaling * tf.math.square(tf.matmul(x, W1) + b1)

# Output Layer Variables:
W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, output_layer], stddev=0.15))
b2 = tf.Variable(tf.zeros([output_layer]))
y = tf.matmul(y1, W2) + b2

# Placeholder for batch of targets:
y_ = tf.placeholder(tf.float32, [None, output_dimension])
```

These are the hyper-parameters used:
```python
class TrainingParameters:
    learning_rate = 0.001
    num_epochs = 70
    num_models = 100
    batch_size = 64
    checkpoint_frequency = 5
```

NOTE:
The time to train data was collected by timing the time to train the first 250 batches. This means that the batch size must be held constant for all future models using the same dataset or the comparison will become invalid.