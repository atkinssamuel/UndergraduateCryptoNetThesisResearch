# Results for the Encrypted Working Model
## Setup
### Encryption Parameters:
```json
{
  "scheme_name": "HE_SEAL",
  "poly_modulus_degree": 8192,
  "security_level": 128,
  "coeff_modulus": [
    60,
    40,
    40,
    60
  ],
  "complex_packing": true
}

```
### Model Details:
#### Hyper-parameters:
Learning Rate = 0.001  
Batch Size = 64  
#### Structure:
One hidden layer with 64 nodes followed by squared activation.
```python
hidden_layer_1 = 64
hidden_layer_2 = 1

# Layer 1 variables:
W1 = tf.Variable(tf.truncated_normal([input_dimension, hidden_layer_1], stddev=0.15))
b1 = tf.Variable(tf.zeros([hidden_layer_1]))
y1 = tf.math.square(tf.matmul(x, W1) + b1)

# Layer 2 variables:
W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
b2 = tf.Variable(tf.zeros([hidden_layer_2]))
y = tf.matmul(y1, W2) + b2
```



## Training Loss, Testing Loss, and Time to Predict for 10 Trials:
|*Trial Number*| Training Loss  | Epoch | Testing Loss | Encrypted Testing Time |
|:-------------:|:----------------:|:-------:|:--------------:|:------------------------:|
| 1            | 13265.905 | 1420 | 4483.858 | 0.113s | 
| 2            | 11995.977 | 1490 | 4461.672 | 0.117s | 
| 3            | 16671.006 | 1490 | 7454.257 | 0.134s | 
| 4            | 19501.011 | 1420 | 5032.806 | 0.135s | 
| 5            | 33521.344 | 1380 | 23845.324 | 0.113s | 
| 6            | 18195.916 | 1400 | 4801.736 | 0.118s | 
| 7            | 11773.683 | 1440 | 5173.491 | 0.112s | 
| 8            | 15020.898 | 1470 | 4826.442 | 0.122s | 
| 9            | 43753.652 | 1490 | 19542.424 | 0.113s | 
| 10           | 10489.166 | 1450 | 4264.462 | 0.113s | 


