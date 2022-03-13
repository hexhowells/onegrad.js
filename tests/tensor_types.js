var onegrad = require("./../onegrad/tensor.js")

var tensor = onegrad.tensor([1,2]);
var ones_tensor = onegrad.ones([2,2])
var zeros_tensor = onegrad.zeros([2,2])
var randn_tensor = onegrad.randn([2,2])
var arange_tensor = onegrad.arange([5])
var arange_tensor2 = onegrad.arange([2, 10])
var eye_tensor = onegrad.eye(3)

console.log("tensor: ", tensor.tolist())
console.log("ones: ", ones_tensor.tolist())
console.log("zeros: ", zeros_tensor.tolist())
console.log("randn: ", randn_tensor.tolist())
console.log("arange (1 arg): ", arange_tensor.tolist())
console.log("arange (2 args): ", arange_tensor2.tolist())
console.log("eye: ", eye_tensor.tolist())