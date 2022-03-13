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

### Tensor Creation
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


### TODO
- ~~implement backprop for all operations~~
- ~~add more optimiser functions~~
- ~~add module class for defining models~~
- add more loss functions (CategoricalCrossEntropy, NLLLoss)
- add more nn abstractions (RNN, LSTM, Conv2d)
- add more activation functions (LeakyReLU, ReLU6, SELU, Softmax)
- add more examples
