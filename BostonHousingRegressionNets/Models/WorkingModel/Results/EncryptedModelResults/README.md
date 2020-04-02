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
| 1            | 24395.377 | 1430 | 9033.983 | 0.058s | 
| 2            | 16078.900 | 1310 | 4949.339 | 0.063s | 
| 3            | 22999.478 | 1450 | 9289.553 | 0.055s | 
| 4            | 10367.337 | 1460 | 5088.867 | 0.060s | 
| 5            | 22604.355 | 1410 | 9960.614 | 0.057s | 
| 6            | 19254.838 | 1450 | 9934.419 | 0.057s | 
| 7            | 12362.853 | 1480 | 7659.735 | 0.061s | 
| 8            | 13741.931 | 1460 | 5143.242 | 0.058s | 
| 9            | 20072.137 | 1460 | 5119.233 | 0.057s | 
| 10           | 13583.890 | 1480 | 3409.236 | 0.062s | 


