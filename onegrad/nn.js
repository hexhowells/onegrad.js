var nj = require("numjs");
var onegrad = require("./tensor.js")

class Linear {
	constructor(inDim, outDim, bias=true) {
		this.inDim = inDim;
		this.outDim = outDim;
		this.useBias = bias
		this.weight = onegrad.randn([outDim, inDim], true);
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


module.exports = {
	Linear,
	MSE
}