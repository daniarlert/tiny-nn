# Tiny NN

This notebooks is a follow through this [tutorial](https://iamtrask.github.io/2015/07/12/basic-python-network/?) written by *iamtrask* about how to implement a bare bones neural network with backpropagation.

## Explanation

A neural network trained with backpropagation will attempt to use input to predict output since backpropagation, in its simplest form, measures statistics like this to make a model.


## Setup


```python
!pip install numpy
```


```python
# import numpy
import numpy as np
```


```python
# seed random numbers to make calculation deterministic
np.random.seed(42)
```

## First part


```python
# define sigmoid function, it gives back a probability
def non_lin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))
```

A sigmoid function maps any value to a value between 0 and 1. It is used to convert numbers to probabilities. It also has several other desirable properties for training neural networks.


```python
# input dataset, each row is a training example
x = np.array(
    [
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
)

# output dataset, each row is a training example
y = np.array([[0, 0, 1, 1]]).T
```


```python
# initialize weights randomly with mean 0
syn_zero = 2 * np.random.random((3, 1)) - 1
```


```python
for i in range(10_000):
    # forward propagation
    l0 = x  # first layer of the network
    l1 = non_lin(np.dot(l0, syn_zero))  # hidden layer

    # misses
    l1_error = y - l1

    # misses multiplied by the slope of the sigmoid at the values of l1
    l1_delta = l1_error * non_lin(l1, True)

    # update weights
    syn_zero += np.dot(l0.T, l1_delta)  # first layer of weights
```


```python
l1
```




    array([[0.00966808],
           [0.00786589],
           [0.99358863],
           [0.99211705]])



## Second part


```python
# update expected output
y = np.array([[0], [1], [1], [0]])
```


```python
# randomly initialize our weights with mean 0
syn_zero = 2 * np.random.random((3, 4)) - 1
syn_one = 2 * np.random.random((4, 1)) - 1
```


```python
for i in range(10_000):
    # feed forward through layers 0, 1 and 2
    l0 = x
    l1 = non_lin(np.dot(l0, syn_zero))
    l2 = non_lin(np.dot(l1, syn_one))

    # misses
    l2_error = y - l2

    if (i % 10_000) == 0:
        print(f"Error: {np.mean(np.abs(l2_error))}")

    # direction of the target value
    l2_delta = l2_error * non_lin(l2, True)

    # how much did each l1 values contribute to the l2 error?
    l1_error = l2_delta.dot(syn_one.T)

    # direction of the target l1
    l1_delta = l1_error * non_lin(l1, True)

    syn_one += l1.T.dot(l2_delta)
    syn_zero += l0.T.dot(l1_delta)
```

    Error: 0.49963945276141286



```python
l2
```




    array([[0.00706947],
           [0.99217766],
           [0.99074812],
           [0.01002659]])


