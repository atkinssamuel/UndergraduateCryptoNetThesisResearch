### Experimental Results:
#### Model 1:
```python
ayer_complexity_growth = 1.5
    l1_scaling, l2_scaling, l3_scaling = 0.00001, 0.00001, 0.00001
    hidden_layer_1 = round(input_dimension * layer_complexity_growth)
    hidden_layer_2 = round(hidden_layer_1 * layer_complexity_growth)
    hidden_layer_3 = round(hidden_layer_2 * layer_complexity_growth)
    output_layer_rescaling_factor = 1000
    output_layer = 1

    # Placeholder for batch of inputs:
    x = tf.placeholder(tf.float32, [None, input_dimension])

    # Layer 1 Variables:
    W1 = tf.Variable(tf.truncated_normal([input_dimension, hidden_layer_1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_layer_1]))
    y1 = l1_scaling * tf.math.square(tf.matmul(x, W1) + b1)

    # Layer 2 Variables:
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
    b2 = tf.Variable(tf.zeros([hidden_layer_2]))
    y2 = l2_scaling * tf.math.square(tf.matmul(y1, W2) + b2)

    # Layer 3 Variables:
    W3 = tf.Variable(tf.truncated_normal([hidden_layer_2, hidden_layer_3], stddev=0.15))
    b3 = tf.Variable(tf.zeros([hidden_layer_3]))
    y3 = l3_scaling * tf.math.square(tf.matmul(y2, W3) + b3)

    # Output Layer Variables:
    W4 = tf.Variable(tf.truncated_normal([hidden_layer_3, output_layer], stddev=0.15))
    b4 = tf.Variable(tf.zeros([output_layer]))
    y = output_layer_rescaling_factor * tf.matmul(y3, W4) + b4

    # Placeholder for batch of targets:
    y_ = tf.placeholder(tf.float32, [None, output_dimension])
```
```python
class TrainingParameters:
    learning_rate = 0.0001
    num_epochs = 70
    num_models = 100
    batch_size = 512
```
###### Result:
- Failed to train
- Initial training loss huge, but not nan
- 0.0% testing accuracy
- Many test outputs were enormous (in the billions of years)
##### Next Model:
- Try something that severely curbs the output of each layer

#### Model 2:
```python
layer_complexity_growth = 1.5
l1_scaling, l2_scaling, l3_scaling = 0.000001, 0.000001, 0.000001
hidden_layer_1 = round(input_dimension * layer_complexity_growth)
hidden_layer_2 = round(hidden_layer_1 * layer_complexity_growth)
hidden_layer_3 = round(hidden_layer_2 * layer_complexity_growth)
output_layer = 1

# Placeholder for batch of inputs:
x = tf.placeholder(tf.float32, [None, input_dimension])

# Layer 1 Variables:
W1 = tf.Variable(tf.truncated_normal([input_dimension, hidden_layer_1], stddev=0.15))
b1 = tf.Variable(tf.zeros([hidden_layer_1]))
y1 = l1_scaling * tf.math.square(tf.matmul(x, W1) + b1)

# Layer 2 Variables:
W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
b2 = tf.Variable(tf.zeros([hidden_layer_2]))
y2 = l2_scaling * tf.math.square(tf.matmul(y1, W2) + b2)

# Layer 3 Variables:
W3 = tf.Variable(tf.truncated_normal([hidden_layer_2, hidden_layer_3], stddev=0.15))
b3 = tf.Variable(tf.zeros([hidden_layer_3]))
y3 = l3_scaling * tf.math.square(tf.matmul(y2, W3) + b3)

# Output Layer Variables:
W4 = tf.Variable(tf.truncated_normal([hidden_layer_3, output_layer], stddev=0.15))
b4 = tf.Variable(tf.zeros([output_layer]))
y = tf.matmul(y3, W4) + b4

# Placeholder for batch of targets:
y_ = tf.placeholder(tf.float32, [None, output_dimension])
```
```python
class TrainingParameters:
    learning_rate = 0.0001
    num_epochs = 70
    num_models = 100
    batch_size = 512
```

##### Result:
- Training very slow
##### Motivation:
- Try the same model with a larger learning rate

#### Model 3:
```python
layer_complexity_growth = 1.5
l1_scaling, l2_scaling, l3_scaling = 0.000001, 0.000001, 0.000001
hidden_layer_1 = round(input_dimension * layer_complexity_growth)
hidden_layer_2 = round(hidden_layer_1 * layer_complexity_growth)
hidden_layer_3 = round(hidden_layer_2 * layer_complexity_growth)
output_layer = 1

# Placeholder for batch of inputs:
x = tf.placeholder(tf.float32, [None, input_dimension])

# Layer 1 Variables:
W1 = tf.Variable(tf.truncated_normal([input_dimension, hidden_layer_1], stddev=0.15))
b1 = tf.Variable(tf.zeros([hidden_layer_1]))
y1 = l1_scaling * tf.math.square(tf.matmul(x, W1) + b1)

# Layer 2 Variables:
W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
b2 = tf.Variable(tf.zeros([hidden_layer_2]))
y2 = l2_scaling * tf.math.square(tf.matmul(y1, W2) + b2)

# Layer 3 Variables:
W3 = tf.Variable(tf.truncated_normal([hidden_layer_2, hidden_layer_3], stddev=0.15))
b3 = tf.Variable(tf.zeros([hidden_layer_3]))
y3 = l3_scaling * tf.math.square(tf.matmul(y2, W3) + b3)

# Output Layer Variables:
W4 = tf.Variable(tf.truncated_normal([hidden_layer_3, output_layer], stddev=0.15))
b4 = tf.Variable(tf.zeros([output_layer]))
y = tf.matmul(y3, W4) + b4

# Placeholder for batch of targets:
y_ = tf.placeholder(tf.float32, [None, output_dimension])
```
```python
class TrainingParameters:
    learning_rate = 0.001
    num_epochs = 70
    num_models = 100
    batch_size = 512
```

##### Result:
- Trained in a stable way for the first 70 epochs, ran again for 150 epochs
- 
#### Model 4:
```python

```
```python

```

##### Result:

#### Model 5:
```python

```
```python

```

##### Result:

#### Model 6:
```python

```
```python

```

##### Result:

#### Model 7:
```python

```
```python

```

##### Result:

#### Model 8:
```python

```
```python

```

##### Result:

#### Model 9:
```python

```
```python

```

##### Result:

#### Model 10:
```python

```
```python

```

##### Result:

#### Model 11:
```python

```
```python

```

##### Result:

#### Model 12:
```python

```
```python

```

##### Result:

