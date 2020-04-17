- Able to achieve reasonable results (testing accuracy = 80%) with a single layer and very few epochs
    - layer_complexity_growth = 1.5
- No issues with testing outputs being restricted and no exploding values observed
- Training plateaued at just 20 epochs
    - Will try a more dense layer (layer_complexity_growth = 2.5)

- With layer_complexity_growth = 2.5 and learning_rate = 0.001, the loss is decreasing non-monotonically. This is strange behaviour because it appears to plateau, but then it continues to decrease.
    - The model finally stopped making noticable improvements at around epoch 65.
    
- With layer_complexity_growth = 2.5 and learning_rate = 0.0005, the loss decreased in a stable and monotonic fashion.
    - The model stopped making noticable improvements at around epoch 55.