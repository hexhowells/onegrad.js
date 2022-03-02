var nj = require("numjs");
var onegrad = require("./tensor.js")

class Linear {
	constructor(inDim, outDim, bias=true) {
		this.inDim = inDim;
		this.outDim = outDim;
		this.useBias = bias
		this.weight = onegrad.randn([inDim, outDim], true);
		if (this.useBias)
			this.bias = onegrad.zeros([outDim], true)
	}

	forward(x) {
		x = x.dot(this.weight);
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


module.exports = {
	Linear
}