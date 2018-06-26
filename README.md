# ecg-loss
Tensorflow implementation for ε-contaminated Gausssian distribution loss with Tensorflow, Keras-compatible.

## Usage

Can be used with Keras using the Tensorflow backend as follows:
```python
from ecg_loss import get_ecg_loss_func

model.compile( 
    loss=get_ecg_loss_func(ecg_c=1.0, ecg_epsilon=0.1),
    optimizer="adam",
    metrics=['mse'])
```

## References
```
J. Tukey, “A survey of sampling from contaminated distributions,”
Contributions to probability and statistics, vol. 2, pp. 448–485, 1960.
```