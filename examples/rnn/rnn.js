var onegrad = require("./../../onegrad/tensor.js");
var optim = require("./../../onegrad/optim.js");
var nn = require("./../../onegrad/nn.js");
var utils = require('./../../onegrad/utils')
var {Loader} = require("./../../datasets/loadText");


// define model
class Model extends nn.Module {
	constructor(inDim, outDim) {
		super()
		this.layers = [
			new nn.RNN(inDim, 128, false),
			new nn.RNN(128, 64, false),
			new nn.Linear(64, outDim, false)
		]
	}

	forward(x) {
		x = this.layers[0].forward(x)
		x = this.layers[1].forward(x)
		x = onegrad.sigmoid(this.layers[2].forward(x))
		return x
	}

	resetPrev() {
		this.layers[0].resetPrev()
		this.layers[1].resetPrev()
	}
}

// hyperparameters
var hp = {
	'lr': 0.001,
	'sampleSize': 16,
	'iterations': 500_000,
	'bs': 8
}

// used for randomChar
var alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

// dataset loader
var loader = new Loader("../../datasets/names.txt", hp.sampleSize)
console.log(`Dataset contains ${loader.data.length} characters and ${loader.vocabSize} unique tokens\n`)

// objects for training
var model = new Model(loader.vocabSize, loader.vocabSize)
var lossfn = new nn.MSE()
var opt = new optim.Adam(model.parameters(), lr=hp.lr, bs=hp.bs)
var encoder = new utils.OnehotEncoder(loader.vocabSize, loader.charsToIdx)


// train network
for (let counter=0; counter<hp.iterations; counter++){
	losses = []
	var inputs = loader.load()
	var firstIndex = inputs[0].indexOf(Math.max(...inputs[0]))

	for (let i=0; i<inputs.length-1; i++) {
		var inputTensor = onegrad.tensor([inputs[i]])
		var out = model.forward(inputTensor)
		
		var nextInput = onegrad.tensor([inputs[i+1]])
		var loss = lossfn.compute(out, nextInput)

		losses.push(loss)
	}

	if (counter%1000 == 0) {
		console.log(`Iter ${counter}`)
		console.log("loss: ", loss.sum().tolist())
		sample()
	}
	if (counter%5000 == 0 && counter != 0) {
		model.save("rnn.json")
	}

	for (loss of losses.reverse()){
		loss.backward()
	}
	model.resetPrev()

	// mini-batch GD
	if (counter % hp.bs) {
		opt.step()
		opt.zeroGrad()
	}
}


function randomChar() {
return alphabet.charAt(Math.floor(Math.random() * 26));
}

// generate sample names from network
function sample(sampleSize=10) {
	for (let i=0; i<sampleSize; i++) {
		var randChar = randomChar()
		var encodedChar = encoder.encode(randChar)
		var inputTensor = onegrad.tensor([encodedChar])
		generatedName = "  " + randChar

		while (true) {
			var out = model.forward(inputTensor)
			var out_arr = out.selection.flatten().tolist()
			var chosenIdx = utils.randomChoice(out_arr, out_arr)
			var idxPred = out_arr.indexOf(chosenIdx[0])

			var char = loader.idxToChars[idxPred]

			if (char == "\n" || generatedName.length > 16)
				break

			generatedName += char
			inputTensor = onegrad.tensor([encoder.encode(char)])
		}
		model.resetPrev()
		console.log(generatedName)
	}
	console.log(`${"-".repeat(50)}\n`)
}
