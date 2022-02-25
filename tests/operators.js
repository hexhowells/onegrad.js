var onegrad = require("./../onegrad/tensor.js")

var a = new onegrad.Tensor([1,2]);
var b = new onegrad.Tensor([3,4]);

console.log("tensor a: ", a.tolist());
console.log("tensor b: ", b.tolist());

console.log("\nOperator Tests");

console.log("\tdot: ", a.dot(b).tolist());
console.log("\tadd: ", a.add(b).tolist());
console.log("\tsub: ", a.sub(b).tolist());
console.log("\tmax: ", a.max().tolist());
console.log("\tmin: ", a.min().tolist());
console.log("\tsum: ", a.sum().tolist());
console.log("\texp: ", a.exp().tolist());