"use strict"
var nj = require("numjs");
var ops = require("./ops.js");
var tensorID = 0


var Tensor = function Tensor(value, op=null, parents=[], requiresGrad=false) {
	this.selection = nj.array(value);
	// when the tensor is printed, the selection (array) is returned
	this[Symbol.for('nodejs.util.inspect.custom')] = () => this.selection;
	this.grad = null;
	this.op = op
	this.parents = [...parents]
	this.shape = this.selection.shape
	this.requiresGrad = requiresGrad
	this.tensorID = tensorID
	tensorID += 1
}

Tensor.prototype.grad = this.grad;

Tensor.prototype.shape = this.shape;

Tensor.prototype.constructDAG = function(graph={nodes:[], edges:[]}) {
	var nodeID = this.tensorID
	var opName = (this.op) ? (this.op.constructor.name) : 'None'

	if (!( graph.nodes.some(el => el.id == nodeID) ))
		graph.nodes.push({id:nodeID, op:opName})

	if (this.parents.length != 0) {
		for (var parent of this.parents) {
			graph.edges.push({from:parent.tensorID, to:nodeID})
			graph = parent.constructDAG(graph)
		}
	}
	return graph
}

Tensor.prototype.backward = function(prev_grad=null) {
	if (!prev_grad || this.grad == null) {
		this.grad = nj.ones(this.selection.shape)
	}

	if (this.parents.length != 0){
		var parent_grads = this.op.backward(...this.parents, this.grad)
		console.assert(Array.isArray(parent_grads), `Error: an op.backward() is not returning an Array`)

		for (let i=0; i < parent_grads.length; i++){
			if(this.parents[i].grad){
				this.parents[i].grad = nj.add(this.parents[i].grad, parent_grads[i])
			} else{
				this.parents[i].grad = parent_grads[i]
			}
		}

		for (const node of this.parents) {
			node.backward(this.grad)
		}
	}
	if (this.requiresGrad == false) {this.grad = null}
}

Tensor.prototype.tolist = function() {
	return this.selection.tolist();
}

Tensor.prototype.dot = function(a) {
	var op = new ops.MatMul()
	return new Tensor(op.forward(this.selection, a.selection), op, [this, a]);
}

Tensor.prototype.mul = function(a) {
	var op = new ops.Multiply()
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

Tensor.prototype.pow = function(a) {
	var op = new ops.Pow()
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

Tensor.prototype.transpose = function() {
	var op = new ops.Transpose()
	return new Tensor(op.forward(this.selection), op, [this]);
}

Tensor.prototype.identity = function() {
	var op = new ops.Identity()
	return new Tensor(op.forward(this.selection), op, [this]);
}

Tensor.prototype.reshape = function(...shape) {
	this.selection = this.selection.reshape(...shape)
	this.shape = this.selection.shape
}

Tensor.prototype.retainGrad = function() {
	this.requiresGrad = true
}


function tensor(data, requiresGrad=true) {
	return new Tensor(data, null, [], requiresGrad)
}

function ones(shape, requiresGrad=true) {
	return new Tensor(nj.ones(shape), null, [], requiresGrad);
}

function zeros(shape, requiresGrad=true) {
	return new Tensor(nj.zeros(shape), null, [], requiresGrad);
}

function randn(shape, requiresGrad=true) {
	return new Tensor(nj.random(shape), null, [], requiresGrad);
}

function arange(args, requiresGrad=true) {
	return new Tensor(nj.arange(...args), null, [], requiresGrad);
}

function eye(shape, requiresGrad=true) {
	return new Tensor(nj.identity(shape), null, [], requiresGrad);
}

function relu(a) {
	var op = new ops.ReLU()
	return new Tensor(op.forward(a.selection), op, [a])
}

function relu6(a) {
	var op = new ops.ReLU6()
	return new Tensor(op.forward(a.selection), op, [a])
}

function leakyRelu(a, negativeSlope) {
	var op = new ops.LeakyReLU(negativeSlope)
	return new Tensor(op.forward(a.selection), op, [a])
}

function sigmoid(a) {
	var op = new ops.Sigmoid()
	return new Tensor(op.forward(a.selection), op, [a])
}

function tanh(a) {
	var op = new ops.Tanh()
	return new Tensor(op.forward(a.selection), op, [a])
}

function selu(a, alpha, lambda) {
	var op = new ops.SELU(alpha, lambda)
	return new Tensor(op.forward(a.selection), op, [a])
}

function softmax(a) {
	var op = new ops.Softmax()
	return new Tensor(op.forward(a.selection), op, [a])
}

module.exports = {
	tensor, 
	ones, 
	zeros,
	randn,
	arange,
	eye,
	relu,
	relu6,
	leakyRelu,
	sigmoid,
	tanh,
	selu,
	softmax
};
