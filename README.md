## About the project

stoch-proc is a library that aims for users to easily define and infer structural time series models in 
[pytorch](https://pytorch.org/) together with [pyro](http://pyro.ai/). `stoch-proc` was previously a submodule in 
[pyfilter](https://github.com/tingiskhan/pyfilter), but was moved into a separate library in order to enable integration
with pyro's inference algorithms.

## Getting started

Follow the below steps in order to install stoch-proc.

### Prerequisites
As mentioned earlier stoch-proc is built on top of pytorch (and pyro). While it's included in the `requirements.txt` 
file, it's **highly recommended** to follow these [instructions](https://pytorch.org/get-started/locally/) to install it 
correctly.

### Installation

stoch-proc is currently not available on PyPi, so you'll have to install it via
```cmd
pip install git+https://github.com/tingiskhan/stoch-proc
```

## Usage

You'll find all the examples [here](./examples), but below you'll find a small example of how to simulate a 
[Lorenz-63 system](https://en.wikipedia.org/wiki/Lorenz_system).

```python
from stochproc import timeseries as ts
import torch
import matplotlib.pyplot as plt


def f(x, s_, r_, b_):
    dxt = s_ * (x.values[..., 1] - x.values[..., 0])
    dyt = r_ * x.values[..., 0] - x.values[..., 1] - x.values[..., 0] * x.values[..., 2]
    dzt = -b_ * x.values[..., 2] + x.values[..., 0] * x.values[..., 1]

    return torch.stack((dxt, dyt, dzt), dim=-1)


initial_values = torch.tensor([-5.91652, -5.52332, 24.5723])

s = 10.0
r = 28.0
b = 8.0 / 3.0

dt = 1e-2
model = ts.RungeKutta(f, (s, r, b), initial_values, dt=dt, event_dim=1)

x = model.sample_states(3_000)
array = x.get_path().numpy()

fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection="3d")

ax.plot3D(array[:, 0], array[:, 1], array[:, 2])
```

Resulting in the following pretty picture

![alt text](./static/lorenz.jpg?raw=true)


## Contributing

Contributions are always welcome! Simply
1. Fork the project.
2. Create your feature branch (I try to follow [Microsoft's naming](https://docs.microsoft.com/en-us/azure/devops/repos/git/git-branching-guidance?view=azure-devops)).
3. Push the branch to origin.
4. Open a pull request.

## License
Distributed under the MIT License, see `LICENSE` for more information.

## Contact
Contact details are located under `setup.py`.