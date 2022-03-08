"use strict"
var nj = require("numjs")


class Optim {

	constructor(params){
		this.params = params;
	}

	zeroGrad() {
		for (var param of this.params) {
			param.grad = null
		}
	}
}

class SGD extends Optim {

	constructor(params, lr, bs=1){
		super(params)
		this.lr = lr;
		this.bs = bs;
	}

	step() {
		for (var param of this.params){
			var batchGrad = nj.divide(param.grad, this.bs)
			var update = nj.multiply(batchGrad, this.lr)
			param.selection = nj.subtract(param.selection, update.T)
		}
	}
}

module.exports = {
	SGD
}