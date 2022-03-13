<p align="center">
  <img src="https://github.com/hexhowells/onegrad.js/blob/main/onegrad-logo.png" width=85%>
</p>

A lightweight deep learning framework in JavaScript. Designed to feel like PyTorch. Uses reverse-mode automatic differentiation and wraps around numjs for base matrix operations.

Examples currently only contain a simple perceptron, however this is still under active development and more sophisticated models are coming soon!

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
Onegrad tensors have multiple properties that can be accessed by the user
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
Onegrad supports most of the common activations functions (with more coming soon!)
```javascript
> var a = onegrad.tensor([-2.5673934, 4]);

> onegrad.sigmoid(a)
tensor([0.07126663626540311, 0.9820137900379085])

> onegrad.tanh(a)
tensor([-0.9882923248658222, 0.999329299739067])

> onegrad.relu(a)
tensor([0, 4])

> var b = onegrad.tensor([[1.3, 5.1, 2.2, 0.7, 1.1]]);
> var softmax = new nn.Softmax();
> softmax.compute(b)
tensor([
  0.02019046473258069,
  0.9025376890165726,
  0.04966052987196014,
  0.011080761983386348,
  0.016530554395500222
])
```

### TODO
- ~~implement backprop for all operations~~
- ~~add more optimiser functions~~
- ~~add module class for defining models~~
- add more loss functions (CategoricalCrossEntropy, NLLLoss)
- add more nn abstractions (LSTM, Conv2d)
- add more activation functions (LeakyReLU, ReLU6, SELU)
- add more examples
- add ability to save and load model weights
