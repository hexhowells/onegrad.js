var onegrad = require("./../onegrad/tensor");
var nn = require("./../onegrad/nn");
var vis = require("./../vis/graph");

// define model
class Model extends nn.Module{
	constructor(inDim, outDim) {
		super()
		this.layers = [
			new nn.Linear(inDim, 128), 
			new nn.Linear(128, outDim, false)
			]
	}

	forward(x) {
		x = onegrad.sigmoid(this.layers[0].forward(x))
		x = onegrad.sigmoid(this.layers[1].forward(x))
		return x
	}
}

// initialise model and loss function
var model = new Model(784, 10);
var lossfn = new nn.MSE();

// create dummy data
var x = onegrad.randn([1, 784], {label: "input"});
var t = onegrad.randn([1, 10], {label: "target"});

// forward pass on dummy data
var out = model.forward(x)
out.label = "out"

var loss = lossfn.compute(out, t)
loss.label = "loss"

// create computational graph and visualise
var dag = loss.constructDAG()
vis.visualise(dag)


