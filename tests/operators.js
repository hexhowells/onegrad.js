var onegrad = require("./../onegrad/tensor.js")
var nn = require("./../onegrad/nn.js");


var a = new onegrad.tensor([1,2]);
var b = new onegrad.tensor([3,4]);
var c = new onegrad.tensor([[-2.5673934, 4]]);
var d = new onegrad.tensor([[0, 1], [2, 3]])
var e = new onegrad.tensor([2])
var f = new onegrad.tensor([[1.3, 5.1, 2.2, 0.7, 1.1]])

console.log("\nInput Tensors")
console.log("\ttensor a: ", a.tolist());
console.log("\ttensor b: ", b.tolist());
console.log("\ttensor c: ", c.tolist());
console.log("\ttensor d: ", d.tolist());
console.log("\ttensor e: ", e.tolist());

console.log("\n--- Operator Tests ---");

console.log("\nBinary Operations")
console.log("\tdot (a, b): ", a.dot(b).tolist());
console.log("\tadd (a, b): ", a.add(b).tolist());
console.log("\tsub (a, b): ", a.sub(b).tolist());
console.log("\tpow (b, e): ", b.pow(e).tolist());

console.log("\nUnary Operations")
console.log("\tmax (a): ", a.max().tolist());
console.log("\tmin (a): ", a.min().tolist());
console.log("\tsum (a): ", a.sum().tolist());
console.log("\texp (a): ", a.exp().tolist());
console.log("\tneg (a): ", a.negative().tolist());
console.log("\tlog (a): ", a.log().tolist());
console.log("\tTranspose (d): ", d.transpose().tolist())

console.log("\nActivation Functions")
console.log("\trelu (c): ", onegrad.relu(c).tolist());
console.log("\tsigmoid (c): ", onegrad.sigmoid(c).tolist());
console.log("\ttanh (c): ", onegrad.tanh(c).tolist());
var softmax = new nn.Softmax()
console.log("\tSoftmax: ", softmax.compute(f).tolist())