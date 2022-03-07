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

### TODO
- ~~implement backprop for all operations~~
- add more loss functions
- add more optimiser functions
- add more nn abstractions
- add more examples
- add module class for defining models
