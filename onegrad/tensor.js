"use strict"
var nj = require("numjs");
//var MatMul = require("./ops.js");


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
	return nj.dot(this.selection, a.selection);
}

Tensor.prototype.add = function(a) {
	return nj.add(this.selection, a.selection)
}

Tensor.prototype.sub = function(a) {
	return nj.subtract(this.selection, a.selection)
}

Tensor.prototype.max = function() {
	return nj.max(this.selection)
}

Tensor.prototype.min = function() {
	return nj.min(this.selection)
}

Tensor.prototype.sum = function() {
	return nj.sum(this.selection)
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
