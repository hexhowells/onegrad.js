var onegrad = require("./../onegrad/tensor.js")


var a = new onegrad.Tensor([1,2]);
var b = new onegrad.Tensor([3,4]);
var c = new onegrad.Tensor([[-2.5673934, 4]]);

console.log("\nInput Tensors")
console.log("\ttensor a: ", a.tolist());
console.log("\ttensor b: ", b.tolist());
console.log("\ttensor c: ", c.tolist());

console.log("\n--- Operator Tests ---");

console.log("\nBinary Operations")
console.log("\tdot: ", a.dot(b).tolist());
console.log("\tadd: ", a.add(b).tolist());
console.log("\tsub: ", a.sub(b).tolist());
console.log("\nUnary Operations")
console.log("\tmax: ", a.max().tolist());
console.log("\tmin: ", a.min().tolist());
console.log("\tsum: ", a.sum().tolist());
console.log("\texp: ", a.exp().tolist());
console.log("\tneg: ", a.negative().tolist());
console.log("\tlog: ", a.log().tolist());
console.log("\nActivation Functions")
console.log("\trelu: ", onegrad.relu(c).tolist());