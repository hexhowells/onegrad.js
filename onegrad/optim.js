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

	constructor(params, lr){
		super(params)
		this.lr = lr;
	}

	step() {
		for (var param of this.params){
			var update = nj.multiply(param.grad, this.lr)
			param.selection = nj.subtract(param.selection, update)
		}
	}
}

module.exports = {
	SGD
}