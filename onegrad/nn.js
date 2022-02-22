var nj = require("numjs");

class Linear {
	constructor(inDim, outDim, bias=true) {
		this.inDim = inDim;
		this.outDim = outDim;
		this.useBias = bias
		this.weight = nj.random([inDim, outDim]);
		if (this.useBias)
			this.bias = nj.zeros([outDim])
	}

	forward(x) {
		x = nj.dot(x, this.weight);
		if (this.useBias)
			x = nj.add(x, this.bias.reshape(1, -1));
		return x;
	}
}


module.exports = Linear;