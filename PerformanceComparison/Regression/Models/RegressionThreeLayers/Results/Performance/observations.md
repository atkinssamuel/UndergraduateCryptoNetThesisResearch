- Training with l1_scaling = 0.00000001, l2_scaling =  0.00000001, l3_scaling =  0.00000001, learning_rate = 0.0005, and layer_complexity_growth = 2.5 for 100 epochs showed that the model was training relatively slow.
    - Will try a larger learning rate
    
- Training with l1_scaling = 0.00000001, l2_scaling =  0.00000001, l3_scaling =  0.00000001, learning_rate = 0.001, and layer_complexity_growth = 2.5 is going slowly, but well after 80 epochs
- Validation loss at 80 epochs = 177273602048.0
    - Optimal validation loss found at epoch 370 (7086777.5)
    - Testing loss = 6754735.5
    
- Trying the same settings but instead of the linear layer scaling, will try implementing a thinner third layer to decrease some of the unnecessary complexity
- Will also be reverting back to the old learning rate

-  Training with l1_scaling = 0.00000001, l2_scaling =  0.00000001, l3_scaling =  0.00000001, learning_rate = 0.0005, layer_complexity_growth = 2.5 and a third layer with 0.5 scaling. Training going smoothly as of epoch 10
    - Minimum found at epoch 105 (7191537.5)
    - Testing loss = 