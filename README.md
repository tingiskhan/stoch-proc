# stoch-proc

`stoch-proc` is a library that aims for users to easily define and infer structural time series models in 
[pytorch](https://pytorch.org/) together with [pyro](http://pyro.ai/). `stoch-proc` was previously integrated in 
[pyfilter](https://github.com/tingiskhan/pyfilter), but was moved into a separate library in order to enable integration
with `pyro`s inference algorithms.

## Installation

`stoch-proc` is currently not available on PyPi, so you'll have to install it via
```cmd
pip install git+https://github.com/tingiskhan/stoch-proc
```

## Example

The below code simulates a Lorenz-63 system utilizing the RK4 schema

```python
from stochproc import timeseries as ts
import torch
import matplotlib.pyplot as plt


def f(x, s_, r_, b_):
    x1 = s_ * (x.values[..., 1] - x.values[..., 0])
    x2 = r_ * x.values[..., 0] - x.values[..., 1] - x.values[..., 0] * x.values[..., 2]
    x3 = -b_ * x.values[..., 2] + x.values[..., 0] * x.values[..., 1]

    return torch.stack((x1, x2, x3), dim=-1)


initial_values = torch.tensor([-5.91652, -5.52332, 24.5723])

s = 10.0
r = 28.0
b = 8.0 / 3.0

dt = 1e-3

model = ts.RungeKutta(f, (s, r, b), initial_values, dt=dt, event_dim=1, num_steps=10)

x = model.sample_path(3_000)
array = x.numpy()

fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection="3d")

ax.plot3D(array[:, 0], array[:, 1], array[:, 2])
```

And we get the following pretty figure

![alt text](./static/lorenz.jpg?raw=true)


You'll find all the examples [here](./examples).