# Results for the Plaintext Working Model
## Setup
### Model Details:
#### Hyper-parameters:
Learning Rate = 0.001  
Batch Size = 64  
#### Structure:
One hidden layer with 64 nodes followed by sigmoid activation.
```python
hidden_layer_1 = 64
hidden_layer_2 = 1

# Layer 1 variables:
W1 = tf.Variable(tf.truncated_normal([input_dimension, hidden_layer_1], stddev=0.15))
b1 = tf.Variable(tf.zeros([hidden_layer_1]))
y1 = tf.math.sigmoid(tf.matmul(x, W1) + b1)

# Layer 2 variables:
W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
b2 = tf.Variable(tf.zeros([hidden_layer_2]))
y = tf.matmul(y1, W2) + b2
```


## Training Loss, Testing Loss, and Time to Predict for 10 Trials:
|*Trial Number*| Training Loss  | Epoch | Testing Loss | Encrypted Testing Time |
|:------------:|:--------------:|:-----:|:------------:|:----------------------:|
| 1            | 3136.269 | 1490 | 2682.773 | 0.055s | 
| 2            | 3352.482 | 1490 | 2993.340 | 0.053s | 
| 3            | 3559.309 | 1490 | 2867.050 | 0.052s | 
| 4            | 3413.542 | 1490 | 2782.768 | 0.062s | 
| 5            | 3432.346 | 1490 | 2838.896 | 0.060s | 
| 6            | 3292.546 | 1490 | 3044.442 | 0.054s | 
| 7            | 3292.656 | 1490 | 2924.610 | 0.061s | 
| 8            | 3169.340 | 1490 | 2835.556 | 0.055s | 
| 9            | 2786.353 | 1490 | 2630.512 | 0.052s | 
| 10           | 3543.082 | 1490 | 2787.062 | 0.053s | 


