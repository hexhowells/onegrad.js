var nj = require("numjs");
var onegrad = require("./tensor.js")

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

class MSE {
	constructor() {
		this.power = new onegrad.tensor([2], false);
	}

	compute(y, yHat) {
		return y.sub(yHat).pow(this.power);
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


module.exports = {
	Linear,
	MSE,
	CrossEntropyLoss
}