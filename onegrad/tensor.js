"use strict"
var nj = require("numjs");
var ops = require("./ops.js");


var Tensor = function Tensor(value, op=null, parents=[]) {
	this.selection = nj.array(value);
	this[Symbol.for('nodejs.util.inspect.custom')] = () => this.selection;
	this.grad = 1;
	this.op = op
	this.parents = [...parents]

}


Tensor.prototype.grad = this.grad;

Tensor.prototype.backward = function() {
	console.log("backward pass not yet implemented");
}


Tensor.prototype.tolist = function() {
	return this.selection.tolist();
}

Tensor.prototype.dot = function(a) {
	var op = new ops.MatMul()
	return new Tensor(op.forward(this.selection, a.selection), op, [this.selection, a]);
}

Tensor.prototype.add = function(a) {
	var op = new ops.Add()
	return new Tensor(op.forward(this.selection, a.selection), op, [this.selection, a]);
}

Tensor.prototype.sub = function(a) {
	var op = new ops.Sub()
	return new Tensor(op.forward(this.selection, a.selection), op, [this.selection, a]);
}

Tensor.prototype.max = function() {
	var op = new ops.Max()
	return new Tensor(op.forward(this.selection), op, [this.selection]);
}

Tensor.prototype.min = function() {
	var op = new ops.Min()
	return new Tensor(op.forward(this.selection), op, [this.selection]);
}

Tensor.prototype.sum = function() {
	var op = new ops.Sum()
	return new Tensor(op.forward(this.selection), op, [this.selection]);
}

/*Tensor.prototype.backward = function(loss) {
	if (this.parent)
		this.parent.backward(loss);
	else
		return this.backward(loss);
}*/


function ones(shape) {
	var tensor = new Tensor();
	tensor.selection = nj.ones(shape);
	return tensor;
}

function zeros(shape) {
	var tensor = new Tensor();
	tensor.selection = nj.zeros(shape);
	return tensor;
}

function randn(shape) {
	var tensor = new Tensor();
	tensor.selection = nj.random(shape);
	return tensor;
}

function arange(...args) {
	var tensor = new Tensor();
	tensor.selection = nj.arange(...args);
	return tensor;
}


module.exports = {
	Tensor, 
	ones, 
	zeros,
	randn,
	arange
};
