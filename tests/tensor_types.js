var onegrad = require("./../onegrad/tensor.js")

var tensor = new onegrad.Tensor([1,2]);
var ones_tensor = new onegrad.ones([2,2])
var zeros_tensor = new onegrad.zeros([2,2])
var randn_tensor = new onegrad.randn([2,2])
var arange_tensor = new onegrad.arange(5)
var arange_tensor2 = new onegrad.arange(2, 10)

console.log("tensor: ", tensor.tolist())
console.log("ones: ", ones_tensor.tolist())
console.log("zeros: ", zeros_tensor.tolist())
console.log("randn: ", randn_tensor.tolist())
console.log("arange (1 arg): ", arange_tensor.tolist())
console.log("arange (2 args): ", arange_tensor2.tolist())