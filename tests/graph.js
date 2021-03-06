var onegrad = require("./../onegrad/tensor.js")

var a = new onegrad.randn([5,5]);
var b = new onegrad.randn([1,5]);
var c = new onegrad.randn([1,5]);

function showParents(parents) {
	for (const p of parents){
		console.log(p.tolist())
	}
}

function showGraph(root) {
	console.log("\nvalue: ", root.tolist())
	console.log("parents: ")
	if (root.parents) showParents(root.parents);
	console.log("op: ", root.op)

	if (!root.parents){
		return null
	}
	for (const node of root.parents) {
		showGraph(node)
	}
}

var d = a.dot(b)
d = onegrad.relu(d)
var loss = d.sub(c).sum()


console.log("----- Computation Graph -----")
showGraph(loss)
