"use strict"
var fs = require('fs');
var nj = require("numjs");
var onegrad = require("./tensor.js")

//
// NN Layers
//
class Linear {
	constructor(inDim, outDim, bias=true) {
		this.inDim = inDim;
		this.outDim = outDim;
		this.useBias = bias
		var nj_arr = nj.random([outDim, inDim]).subtract(0.5)
		this.weight = onegrad.tensor(nj_arr)
		if (this.useBias)
			this.bias = onegrad.zeros([outDim, 1], true)
	}

	forward(x) {
		x = this.weight.dot(x)
		if (this.useBias)
			x = x.add(this.bias);
		return x;
	}

	parameters() {
		if (this.useBias) {
			return [this.weight, this.bias]
		} else {
			return [this.weight]
		}
	}
}


class RNN {
	constructor(inDim, outDim, bias=true) {
		this.inDim = inDim;
		this.outDim = outDim;
		this.useBias = bias

		if (this.useBias)
			this.bias = onegrad.zeros([outDim, 1], true)

		var w_arr = nj.random([outDim, inDim]).subtract(0.5)
		var hw_arr = nj.random([outDim, outDim]).subtract(0.5)

		this.weight = onegrad.tensor(w_arr)
		this.hiddenWeight = onegrad.tensor(hw_arr)
		this.prevOutput = onegrad.zeros([1, outDim])
	}

	forward(x) {
		x = this.weight.dot(x)
		var prev = this.hiddenWeight.dot(this.prevOutput)
		x = x.add(prev)

		if (this.useBias)
			x = x.add(this.bias);

		x = onegrad.tanh(x)
		this.prevOutput = x
		return x
	}

	parameters() {
		if (this.useBias) {
			return [this.weight, this.hiddenWeight, this.bias]
		} else {
			return [this.weight, this.hiddenWeight]
		}
	}

	resetPrev() {
		this.prevOutput = onegrad.zeros([1, this.outDim])
		this.hiddenWeight.parents = []
	}
}

//
// Model Parent Class
//
class Module {
	constructor() {
		this.train = true
	}

	parameters() {
		var modelParams = []
		for (var layer of this.layers){
			for (var param of layer.parameters()) {
				modelParams.push(param)
			}
		}
		return modelParams
	}

	train() {
		this.train = true
	}

	eval() {
		this.train = false
	}

	save(path) {
		var params = this.parameters()
	    var array = []
	    for (var param of params) {
	      array.push(param.selection.tolist())
	    }
	    fs.writeFileSync(path, JSON.stringify(array));
	}

	load(path) {
		var params = this.parameters()
	    const fileContent = fs.readFileSync(path);
	    const array = JSON.parse(fileContent);
	    for (let i=0; i<params.length; i++) {
	    	params[i].selection = nj.array(array[i])
	    }
	}
}

//
// Loss Functions
//
class MSE {
	constructor() {
		this.power = new onegrad.tensor([2], false);
	}

	compute(y, yHat) {
		return y.sub(yHat).pow(this.power);
	}
}

class MAE {
	constructor() {
		this.power = new onegrad.tensor([2], false);
		this.half = new onegrad.tensor([0.5], false)
	}

	compute(y, yHat) {
		return y.sub(yHat).pow(this.power).pow(this.half);
	}
}

class CrossEntropyLoss {
	constructor() {
		this.one = onegrad.tensor([1]);
	}

	compute(y, yHat) {
		var comp1 = y.dot(yHat.log());
		var comp2 = y.negative().add(this.one);
		var comp3 = yHat.negative().add(this.one);
		var comp4 = comp3.log();
		var loss = comp1.add((comp2.dot(comp4))).negative();
		return loss;
	}
}

//
// Activation Functions
//
class Softmax {
	constructor() {
		this.divPower = onegrad.tensor([-1])
	}

	compute(x) {
		var denom = x.exp().transpose().sum()
		denom.reshape([1,1])
		var arr =  x.exp().transpose().dot(denom.pow(this.divPower))
		return arr
	}

	parameters() {
		return []
	}
}


module.exports = {
	Linear,
	RNN,
	Module,
	MSE,
	MAE,
	CrossEntropyLoss,
	Softmax
}