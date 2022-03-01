#  Onegrad.js

A lightweight javascript machine learning library inspired by PyTorch. wraps around numjs for base matrix operations.

### Example
```javascript
var onegrad = require("./../onegrad/tensor.js")

var x = new onegrad.eye(3);
var y = new onegrad.Tensor([[2, 0, -2]]);

var z = y.dot(x).sum();

z.backward();

console.log("x: ", x.grad.tolist());  // dz/dx
console.log("y: ", y.grad.tolist());  // dz/dy
```
