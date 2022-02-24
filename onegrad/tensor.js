"use strict"
var nj = require("numjs");
var ops = require("./ops.js");


var Tensor = function Tensor() {
	this.selection = nj.array(arguments[0]);
	this[Symbol.for('nodejs.util.inspect.custom')] = () => this.selection;
	this.savedTensors = [];
	this.grad = 1;
}


Tensor.prototype.grad = this.grad;


Tensor.prototype.tolist = function() {
	return this.selection.tolist();
}

Tensor.prototype.dot = function(a) {
	var op = new ops.MatMul()
	return op.forward(this.selection, a.selection);
	return nj.dot(this.selection, a.selection);
}

Tensor.prototype.add = function(a) {
	var op = new ops.Add()
	return op.forward(this.selection, a.selection);
	return nj.dot(this.selection, a.selection);
}

Tensor.prototype.sub = function(a) {
	var op = new ops.Sub()
	return op.forward(this.selection, a.selection);
	return nj.dot(this.selection, a.selection);
}

Tensor.prototype.max = function() {
	var op = new ops.Max()
	return op.forward(this.selection);
	return nj.dot(this.selection, a.selection);
}

Tensor.prototype.min = function() {
	var op = new ops.Min()
	return op.forward(this.selection);
	return nj.dot(this.selection, a.selection);
}

Tensor.prototype.sum = function() {
	var op = new ops.Sum()
	return op.forward(this.selection);
	return nj.dot(this.selection, a.selection);
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
