<p align="center">
  <img src="https://github.com/hexhowells/onegrad.js/blob/main/onegrad-logo.png" width=85%>
</p>

A lightweight deep learning framework in JavaScript. Designed to feel like PyTorch. Uses reverse-mode automatic differentiation and wraps around numjs for base matrix operations.

See ```Examples/``` to see what onegrad can do.

### Example
```javascript
var onegrad = require("./../onegrad/tensor.js")

var x = new onegrad.eye(3);
var y = new onegrad.tensor([[2, 0, -2]]);

var z = y.dot(x).sum();

z.backward();

console.log("x: ", x.grad.tolist());  // dz/dx
console.log("y: ", y.grad.tolist());  // dz/dy
```
***

## Tensor Creation
Onegrad has many helper functions for creating tensors. All helper functions have the optional argument ```requiresGrad``` for specifying if the tensors gradient should be saved after a backwards pass (set to ```true``` by default)
```javascript
> onegrad.tensor([1, 2, 3]);
tensor([1, 2, 3])

> onegrad.ones([2, 2]);
tensor([[1, 1], 
        [1, 1]])

> onegrad.zeros([2, 2]);
tensor([[0, 0], 
        [0, 0]])

> onegrad.randn([2]);
tensor([0.5607564512157093, 0.9575847907431982])

> onegrad.arange([5]);
tensor([0, 1, 2, 3, 4])

> onegrad.arange([5, 10]);
tensor([5, 6, 7, 8, 9])

> onegrad.eye(3);
tensor([[ 1, 0, 0 ], 
        [ 0, 1, 0 ], 
        [ 0, 0, 1 ]])

```
## Infomation about the Tensor
Onegrad tensors have multiple properties that can be accessed by the user:
- ```.shape``` dimenstions of the tensor
- ```.selection``` Numjs array contained in the tensor
- ```.grad``` gradient of the tensor
- ```.op``` operation used to create the tensor (set to none when made with the above helper functions)
- ```.parents``` parent tensors used for the tensors creation (empty array for tensors made with the above helpers)
- ```requiresGrad``` whether the tensor will save its gradient value after each backwards pass

Additionaly ```.tolist()``` is used to convert the tensor into a js array.
```javascript
> var a = onegrad.tensor([1, 2, 3]);
> a
tensor([1, 2, 3])

> a.tolist();
[1, 2, 3]

> a.shape
[3]

> a.selection
array([1, 2, 3])

> a.requiresGrad
true
```
Calling ```.backward()``` on a tensor performs a backwards pass on its DAG.
```javascript
> var a = onegrad.tensor([1, 2, 3]);
> var b = onegrad.tensor([2, 2, 2]);
> var c = a.sub(b);
> c
tensor([-1, 0, 1])

> c.parents
[tensor([1, 2, 3]), tensor([2, 2, 2])

> c.op
Sub

> c.backward();
> a.grad
tensor([1, 1, 1])
> b.grad
tensor([-1, -1, -1])

```
## Tensor Operations
Onegrad supports most tensor operations required for deep learning

#### Unary operations
```javascript
> var a = onegrad.tensor([1, 2]);

> a.max()
tensor([2]

> a.min()
tensor([1])

> a.sum()
tensor([3])

> a.exp()
tensor([2.718281828459045, 7.38905609893065])

> a.negative()
tensor([-1, -2])

> a.log()
tensor([0, 0.300975762292638])
```

#### Binary Operations
```javascript
> var a = onegrad.tensor([1, 2]);
> var b = onegrad.tensor([3, 4]);
> var c = onegrad.tensor([2]);

> a.dot(b)
tensor([11])

> a.add(b)
tensor([4, 6])

> a.sub(b)
tensor([-2, -2])

> a.pow(b)
tensor([1, 4])
```

## Tensor Manipulation
Onegrad supports the ```transpose``` and ```reshape``` operations for manipulating tensors.

**Note:** ```reshape``` reshapes the tensor **in-place**
```javascript
> var a = onegrad.tensor([[1, 2, 3, 4]]);
> a.shape
[1, 4]

> a.transpose()
tensor([[1], [2], [3], [4]])
> a.transpose().shape
[4, 1]

> a.reshape([2, 2])
> a
tensor([[1, 2], 
        [3, 4]])
> a.shape
[2, 2]

```

## Activation Functions
Onegrad supports most of the common activations functions
```javascript
> var a = onegrad.tensor([-2.5673934, 4]);

> onegrad.sigmoid(a)
tensor([0.07126663626540311, 0.9820137900379085])

> onegrad.tanh(a)
tensor([-0.9882923248658222, 0.999329299739067])

> onegrad.relu(a)
tensor([0, 4])

> onegrad.relu6(a)
tensor([0, 4])

> onegrad.leakyRelu(a)
tensor([-0.025673934, 4])

> onegrad.selu(a)
tensor([-1.5448988171423363, 4])

> var b = onegrad.tensor([[1.3, 5.1, 2.2, 0.7, 1.1]]);
> onegrad.softmax(b)
tensor([
  0.02019046473258069,
  0.9025376890165726,
  0.04966052987196014,
  0.011080761983386348,
  0.016530554395500222
])
```

## Layer Abstractions
Onegrad supports some layer abstractions to help make building networks easier.
All layers have 3 parameters:
- ```inDim``` number of input nodes
- ```outDim``` number of output nodes
- ```useBias``` bias toggle (true by default)

**Note:** recurrent layers requires a call to ```.resetPrev()``` to reset the previous hidden output.
```javascript
> var x = onegrad.randn([1, 10]);

> var denselayer = new nn.Linear(10, 1, false);
> denseLayer.forward(x)
tensor([0.4736675307891193])

> var rnnLayer = new nn.RNN(10, 1, false);
> rnnLayer.forward(x)
tensor([0.06440360973284968])

// reset previous hidden output after each complete forward pass on a sequence
> rnnLayer.resetPrev()
```
The parameters of each layer can be accessed using the ```.parameters()``` function.
```javascript
> var rnnLayer = new nn.RNN(10, 1, true);
> rnnLayer.parameters()
list([tensor([...]), tensor([...]), tensor([...])])
```

## Modules
Modules can be used to define entire models inside a class by extending from ```nn.Modules```.
Defined models requires ```constructor()``` for defining the model layers and ```forward(x)``` for specifying how the layers interact.

Model layers need to be placed in an array called ```layers``` (required for the framework to extract model parameters)
```javascript
> class Model extends nn.Module {
    constructor(inDim, outDim) {
      super()
      this.layers = [
          new nn.Linear(inDim, 100),
          new nn.Linear(100, outDim)
      ]
    }
    
    forward(x) {
      x = onegrad.sigmoid(this.layers[0].forward(x))
      x = onegrad.sigmoid(this.layers[1].forward(x))
      return x
    }
  }
> var model = new Model(10, 1);
> var x = onegrad.randn([1, 10]);
> model.forward(x)
tensor([0.7826402419856238])

```

### Save and Load Models
Models can be saved and loaded from a .json file. requires the model filepath to be specified.
```javascript
> model.save('model.json');
> model.load('model.json');
```

## Loss Functions
Onegrad supports a few of the basic loss functions.

To compute the loss call ```.compute(output, target)``` on the loss function.
```javascript
> var x = onegrad.randn([1, 10]);
> var tar = onegrad.randn([1, 10]);

> var lossfn = new nn.MSE();
> lossfn.compute(x, tar)
tensor([0.0270..., 0.2257..., 0.0173..., 0.4238..., 0.2901...])

> var lossfn = new nn.MAE();
> lossfn.compute(x, tar)
tensor([0.1082..., 0.0471..., 0.4704..., 0.4681..., 0.0965...])
```

## Optimisers
Onegrad currently supports the SGD and Adam optimisers.

Parameters of ```SGD```
- ```params``` parameters to update
- ```lr``` learning rate (default 0.01)
- ```bs``` batch size (default 1)

Parameters of ```Adam```
- ```params``` parameters to update
- ```lr``` learning rate (default 0.001)
- ```bs``` batch size
- ```b1``` beta 1 (default 0.9)
- ```b2``` beta 2 (default 0.999)
- ```eps``` epsilon (default 1e-8)

```javascript
> var opt = new optim.SGD(model.parameters(), lr=0.01);

// update model weights
> opt.step()

// reset parameter gradients
> opt.zeroGrad()
```

### Gradient Decay
Onegrad supports a basic learning rate scheduler which decays the learning rate every *n* steps.

Parameters
- ```optim``` optimiser to schedule
- ```stepSize``` how many steps to decay on (default 30)
- ```gamma``` how much to decay the gradient (default 0.1)
- ```lastEpoch``` the index of last epoch (default -1)

```javascript
> var opt = new optim.SGD(model.parameters(), lr=0.01);
> var scheduler = new optim.StepLR(opt);

// step scheduler every iteration
scheduler.step()
```

### TODO
- ~~implement backprop for all operations~~
- ~~add more optimiser functions~~
- ~~add module class for defining models~~
- ~~add more activation functions (LeakyReLU, ReLU6, SELU)~~
- ~~add ability to save and load model weights~~
- add more loss functions (CategoricalCrossEntropy, NLLLoss)
- add more nn abstractions (LSTM, GRU, Conv2d)
- add more examples
- create visualisations for the DAG
- webpack source code
