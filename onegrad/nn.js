"use strict"
var fs = require('fs');
var nj = require("numjs");
var onegrad = require("./tensor.js")

//
// NN Layers
//
class Linear {
	constructor(inDim, outDim, bias=false) {
		this.inDim = inDim;
		this.outDim = outDim;
		this.useBias = bias
		var nj_arr = nj.random([outDim, inDim]).subtract(0.5)
		this.weight = onegrad.tensor(nj_arr, {label:"Linear: weight"})
		if (this.useBias)
			this.bias = onegrad.zeros([outDim, 1], {label:"Linear: bias"})
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
	constructor(inDim, outDim, bias=false) {
		this.inDim = inDim;
		this.outDim = outDim;
		this.useBias = bias

		if (this.useBias)
			this.bias = onegrad.zeros([outDim, 1], {label:"RNN: bias"})

		var w_arr = nj.random([outDim, inDim]).subtract(0.5)
		var hw_arr = nj.random([outDim, outDim]).subtract(0.5)

		this.weight = onegrad.tensor(w_arr, {label:"RNN: weight"})
		this.hiddenWeight = onegrad.tensor(hw_arr, {label:"RNN: hidden weight"})
		this.prevOutput = onegrad.zeros([1, outDim], {label:"RNN: prev"})
	}

	forward(x) {
		x = this.weight.dot(x)
		var prev = this.hiddenWeight.dot(this.prevOutput)
		x = x.add(prev)

		if (this.useBias)
			x = x.add(this.bias);

		x = onegrad.tanh(x)
		this.prevOutput = x.identity()
		this.prevOutput.parents = []
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

class GRU {
	constructor(inDim, outDim) {
		this.inDim = inDim;
		this.outDim = outDim;
		
		this.one = onegrad.ones([1])

		var wz_arr = nj.random([outDim, inDim]).subtract(0.5)
		var uz_arr = nj.random([outDim, outDim]).subtract(0.5)

		var wr_arr = nj.random([outDim, inDim]).subtract(0.5)
		var ur_arr = nj.random([outDim, outDim]).subtract(0.5)

		var wh_arr = nj.random([outDim, inDim]).subtract(0.5)
		var uh_arr = nj.random([outDim, outDim]).subtract(0.5)

		var wy_arr = nj.random([outDim, outDim]).subtract(0.5)

		this.wz = onegrad.tensor(wz_arr, {label:"GRU: wz"})
		this.uz = onegrad.tensor(uz_arr, {label:"GRU: uz"})

		this.wr = onegrad.tensor(wr_arr, {label:"GRU: wr"})
		this.ur = onegrad.tensor(ur_arr, {label:"GRU: ur"})

		this.wh = onegrad.tensor(wh_arr, {label:"GRU: wh"})
		this.uh = onegrad.tensor(uh_arr, {label:"GRU: uh"})

		this.wy = onegrad.tensor(wy_arr, {label:"GRU: wy"})
		
		this.prev = onegrad.zeros([1, outDim], {label:"GRU: prev"})
	}

	forward(x) {
		// reset gates
		var zt = onegrad.sigmoid( this.wz.dot(x).add(this.uz.dot(this.prev)) )
		var rt = onegrad.sigmoid( this.wr.dot(x).add(this.ur.dot(this.prev)) )

		// hidden units
		var hHatTemp = rt.mul(this.prev)
		
		var hHat = onegrad.tanh( this.wh.dot(x).add(this.uh.dot(hHatTemp)) )

		var hTemp = zt.negative().add(this.one)
		var h = zt.mul(this.prev).add(hTemp.mul(hHat))

		// output unit
		var y = this.wy.dot(h)
		y = onegrad.sigmoid(y)
		
		this.prev = y.identity()
		//this.prev.parents = []

		return y
	}

	parameters() {
		return [
			this.wz,
			this.uz,
			this.wr,
			this.uz,
			this.wh,
			this.uz,
			this.wy
		]
	}

	resetPrev() {
		this.prev = onegrad.zeros([1, this.outDim])
		this.wz.parents = []
		this.uz.parents = []
		this.wr.parents = []
		this.uz.parents = []
		this.wh.parents = []
		this.uz.parents = []
		this.wy.parents = []
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


module.exports = {
	Linear,
	RNN,
	GRU,
	Module,
	MSE,
	MAE,
	CrossEntropyLoss
}