var onegrad = require("./../onegrad/tensor.js")

var x = new onegrad.eye(3);
var y = new onegrad.tensor([[2, 0, -2]]);

var z = y.dot(x).sum();
z.retainGrad()

z.backward();

console.log("---- Inputs ----");
console.log(x.tolist());
console.log(y.tolist());

console.log("\n---- Grads ----");
console.log("x: ", x.grad.tolist());
console.log("y: ", y.grad.tolist());
console.log("z: ", z.grad.tolist());