var onegrad = require("./../onegrad/tensor.js");
var optim = require("./../onegrad/optim.js");
var nn = require("./../onegrad/nn.js");
var {loadMnist} = require("./../datasets/loadMnist")


// define model
class Model extends nn.Module{
	constructor(inDim, outDim) {
		super()
		this.layers = [
			new nn.Linear(inDim, 200, false), 
			new nn.Linear(200, 128, false), 
			new nn.Linear(128, outDim, false)]
	}

	forward(x) {
		x = onegrad.sigmoid(this.layers[0].forward(x))
		x = onegrad.sigmoid(this.layers[1].forward(x))
		x = onegrad.sigmoid(this.layers[2].forward(x))
		return x
	}
}

// hyperparameters
const epochs = 10;
var lr = 0.001;
var batchSize = 32;

// create objects for training
var model = new Model(784, 10);
var opt = new optim.Adam(model.parameters(), lr=lr, bs=batchSize);
var lossfn = new nn.MSE();

// load dataset and start training
var [training, testing] = loadMnist(batchSize)
console.log(`\nloaded ${training.labels.length * batchSize} images`)
train(training, testing)


function train(training, testing) {
	for (let epoch=0; epoch<epochs; epoch++){
		console.log(`\nEpoch ${epoch+1} of ${epochs}`);

		for (let i=0; i<training.data.length; i++){
			var input_tensor = training.data[i];
			var target_tensor = training.labels[i].transpose();

			var output = model.forward(input_tensor.transpose())

			var loss = lossfn.compute(target_tensor, output);

			loss.backward()
			opt.step();
			opt.zeroGrad()
		}
		console.log(`Accuracy: ${evaluate(training.data, training.labels)}%`)
		console.log(`Val Accuracy: ${evaluate(testing.data, testing.labels)}%`)
	}
}


function evaluate(images, labels) {
	var acc = 0

	for (let i=0; i<images.length; i++) {
		// forward pass
		model.eval()
		var output = model.forward(images[i].transpose())
		output = output.transpose()
		
		// extract answer
		var out_arr = output.selection.tolist()
		var label_arr = labels[i].selection.tolist()
		for (let i=0; i<out_arr.length; i++){
			var out_ans = out_arr[i].indexOf(Math.max(...out_arr[i]))
			var tar_ans = label_arr[i].indexOf(Math.max(...label_arr[i]))

			if (out_ans == tar_ans) {
				acc += 1
			}
		}
	}
	return ((acc / (images.length*batchSize) ) * 100).toFixed(2)
}
