- Takes significantly longer (more epochs) to get reasonable results
- Aggressive (0.00000001) layer scaling showing promise


- Training with l1_scaling = 0.00000001, learning_rate = 0.0005, and layer_complexity_growth = 10 for 200 epochs showed no signs of overtraining. Could potentially train model for even longer.
    - Testing output plot snippet with the aformentioned settings appears to be confined between 1200 and 2600.
    - Will try to train for 1k epochs.

- Training with l1_scaling = 0.00000001, learning_rate = 0.0005, and layer_complexity_growth = 2.5 for 500 epochs showed training plateauing at around 300 epochs. 
    - Testing output produced an accuracy of 80.97% which is comparable to the plaintext model