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
	return new Tensor(op.forward(this.selection, a.selection), op, [this, a]);
}

Tensor.prototype.add = function(a) {
	var op = new ops.Add()
	return new Tensor(op.forward(this.selection, a.selection), op, [this, a]);
}

Tensor.prototype.sub = function(a) {
	var op = new ops.Sub()
	return new Tensor(op.forward(this.selection, a.selection), op, [this, a]);
}

Tensor.prototype.max = function() {
	var op = new ops.Max()
	return new Tensor(op.forward(this.selection), op, [this]);
}

Tensor.prototype.min = function() {
	var op = new ops.Min()
	return new Tensor(op.forward(this.selection), op, [this]);
}

Tensor.prototype.sum = function() {
	var op = new ops.Sum()
	return new Tensor(op.forward(this.selection), op, [this]);
}

Tensor.prototype.exp = function() {
	var op = new ops.Exp()
	return new Tensor(op.forward(this.selection), op, [this]);
}

Tensor.prototype.negative = function() {
	var op = new ops.Negative()
	return new Tensor(op.forward(this.selection), op, [this]);
}

Tensor.prototype.log = function() {
	var op = new ops.Log()
	return new Tensor(op.forward(this.selection), op, [this]);
}

/*Tensor.prototype.backward = function(loss) {
	if (this.parent)
		this.parent.backward(loss);
	else
		return this.backward(loss);
}*/


function ones(shape) {
	return new Tensor(nj.ones(shape));
}

function zeros(shape) {
	return new Tensor(nj.zeros(shape));
}

function randn(shape) {
	return new Tensor(nj.random(shape));
}

function arange(...args) {
	return new Tensor(nj.arange(...args));
}

function relu(a) {
	var op = new ops.ReLU()
	return new Tensor(op.forward(a.selection), op, [a])
}


module.exports = {
	Tensor, 
	ones, 
	zeros,
	randn,
	arange,
	relu
};
