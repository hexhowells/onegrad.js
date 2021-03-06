var onegrad = require("./../onegrad/tensor.js")
var nn = require("./../onegrad/nn.js");


var a = onegrad.tensor([1,2]);
var b = onegrad.tensor([3,4]);
var c = onegrad.tensor([-2.5673934, 4, 8.2]);
var d = onegrad.tensor([[0, 1], [2, 3]])
var e = onegrad.tensor([2])
var f = onegrad.tensor([[1.3, 5.1, 2.2, 0.7, 1.1]])

console.log("\nInput Tensors")
console.log("\ttensor a: ", a.tolist());
console.log("\ttensor b: ", b.tolist());
console.log("\ttensor c: ", c.tolist());
console.log("\ttensor d: ", d.tolist());
console.log("\ttensor e: ", e.tolist());
console.log("\ttensor f: ", f.tolist());

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
console.log("\tSoftmax (f): ", onegrad.softmax(f).tolist())
console.log("\trelu6 (c): ", onegrad.relu6(c).tolist());
console.log("\tleakyRelu (c): ", onegrad.leakyRelu(c).tolist());
console.log("\tselu (c): ", onegrad.selu(c).tolist());