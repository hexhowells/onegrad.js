"use strict"
var nj = require("numjs");
var ops = require("./ops.js");


var Tensor = function Tensor(value, op=null, parents=[]) {
	this.selection = nj.array(value);
	this[Symbol.for('nodejs.util.inspect.custom')] = () => this.selection;
	this.grad = null;
	this.op = op
	this.parents = [...parents]

}


Tensor.prototype.grad = this.grad;

Tensor.prototype.backward = function(prev_grad=null) {
	if (!prev_grad) {
		this.grad = nj.ones(this.selection.shape)
	}

	if (this.parents.length == 0){
		return 0
	}

	var parent_grads = this.op.backward(...this.parents, this.grad)
	for (let i=0; i < parent_grads.length; i++){
		this.parents[i].grad = parent_grads[i]
	}
	
	for (const node of this.parents) {
		node.backward(this.grad)
	}
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

function eye(shape) {
	return new Tensor(nj.identity(shape));
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
	eye,
	relu
};
