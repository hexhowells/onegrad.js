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

class StepLR {
	constructor(optimiser, stepSize=30, gamma=0.1, lastEpoch){
		this.opt = optimiser;
		this.stepSize = stepSize;
		this.gamma = gamma;
		this.lastEpoch = lastEpoch;
	}

	step() {
		this.lastEpoch += 1
		if (this.lastEpoch % stepSize == 0 && this.lastEpoch != 0) {
			this.opt.lr *= this.gamma
		}
	}
}

module.exports = {
	SGD,
	StepLR
}