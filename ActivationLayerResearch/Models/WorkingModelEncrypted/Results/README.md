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
|*Trial Number*| Training Loss | Epoch | Testing Loss | Encrypted Testing Time |
|:------------:|:---------:|:----:|:--------:|:------:|
| 1            | 28066.324 | 1450 | 9339.782 | 0.192s | 
| 2            | 11502.458 | 1450 | 4414.307 | 0.212s | 
| 3            | 23530.930 | 1490 | 9249.729 | 0.187s | 
| 4            | 10367.337 | 1470 | 5286.291 | 0.204s | 
| 5            | 14402.311 | 1320 | 5973.787 | 0.194s | 
| 6            |  7523.974 | 1350 | 3440.328 | 0.181s | 
| 7            | 14092.383 | 1430 | 5007.370 | 0.190s | 
| 8            | 22448.347 | 1420 |12125.124 | 0.176s | 
| 9            | 26273.113 | 1460 | 9009.018 | 0.181s | 
| 10           | 26207.563 | 1440 | 9409.127 | 0.187s | 


