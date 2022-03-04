var onegrad = require("./../onegrad/tensor.js")
var optim = require("./../onegrad/optim.js")
var nn = require("./../onegrad/nn.js")

// model
var layer = new nn.Linear(3, 1, false)

// training data
var inputs = [[0,0,1], [1,1,1], [1,0,1], [0,1,1]];
var targets = [0, 1, 1, 0];

// hyperparameters
const epochs = 250;
var lr = 1

var opt = new optim.SGD(layer.parameters(), lr=lr);
var lossfn = new nn.MSE()

// training
for (let epoch=0; epoch<epochs; epoch++){
	console.log(`\nEpoch ${epoch+1} of ${epochs}`);

	for (let i=0; i<inputs.length; i++){
		var input_tensor = new onegrad.tensor([inputs[i]]);
		var target_tensor = new onegrad.tensor([[targets[i]]]);

		var output = onegrad.sigmoid(layer.forward(input_tensor));
		var loss = lossfn.compute(target_tensor, output);
		
		console.log("out: ", output.tolist());
	
		opt.zeroGrad();
		loss.backward();
		opt.step();
	}
}

// testing
var test_input = new onegrad.tensor([[1,0,0]]);
var output = onegrad.sigmoid(layer.forward(test_input));
console.log("\ntest input [1,0,0] -> ", output.tolist());