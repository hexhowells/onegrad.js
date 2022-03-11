var onegrad = require("./../onegrad/tensor.js");
var optim = require("./../onegrad/optim.js");
var nn = require("./../onegrad/nn.js");
var {loadMnist} = require("./../datasets/loadMnist")


// model
var layer1 = new nn.Linear(784, 200, false)
var layer2 = new nn.Linear(200, 128, false)
var layer3 = new nn.Linear(128, 10, false)

// hyperparameters
const epochs = 10;
var lr = 0.001;
var batchSize = 32;

var params = layer1.parameters().concat(layer2.parameters()).concat(layer3.parameters());
var opt = new optim.Adam(params, lr=lr, bs=batchSize);
var lossfn = new nn.MSE();
//var scheduler = new optim.StepLR(opt, stepSize=5, gamma=0.1)


function evaluate(images, labels) {
	var acc = 0

	for (let i=0; i<images.length; i++) {
		// forward pass
		var l1_output = onegrad.sigmoid(layer1.forward(images[i].transpose()));
		var l2_output = onegrad.sigmoid(layer2.forward(l1_output));
		var l3_output = onegrad.sigmoid(layer3.forward(l2_output));
		l3_output = l3_output.transpose()
		
		// extract answer
		var out_arr = l3_output.selection.tolist()
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



var [training, testing] = loadMnist(batchSize)
console.log(`\nloaded ${training.labels.length * batchSize} images`)

// training
for (let epoch=0; epoch<epochs; epoch++){
	console.log(`\nEpoch ${epoch+1} of ${epochs}`);

	for (let i=0; i<training.data.length; i++){
		var input_tensor = training.data[i];
		var target_tensor = training.labels[i].transpose();

		var l1_output = onegrad.sigmoid(layer1.forward(input_tensor.transpose()));
		var l2_output = onegrad.sigmoid(layer2.forward(l1_output));
		var l3_output = onegrad.sigmoid(layer3.forward(l2_output));

		var loss = lossfn.compute(target_tensor, l3_output);

		loss.backward()
		opt.step();
		opt.zeroGrad()
		//scheduler.step()
	}
	console.log(`Accuracy: ${evaluate(training.data, training.labels)}%`)
	console.log(`Val Accuracy: ${evaluate(testing.data, testing.labels)}%`)
}

