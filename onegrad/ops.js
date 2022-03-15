"use strict"
var nj = require("numjs")


// ----- Binary Operations -----

class MatMul {

	forward(a, b) {
		return nj.dot(b, a.T)
	}

	backward(a, b, prev_grad) {
		var grad_weight = nj.dot(prev_grad, a.selection);
		var grad_input = nj.dot(b.selection.T, prev_grad)

		return [grad_input, grad_weight]
	}
}

class Add {

	forward(a, b) {
		if (b.shape == 1){
			b = b.get(0)
		}
		
		return nj.add(a, b)
	}

	backward(a, b, prev_grad) {
		return [prev_grad, prev_grad]
	}
}

class Sub {

	forward(a, b) {
		return nj.subtract(a, b)
	}

	backward(a, b, prev_grad) {
		return [prev_grad, nj.negative(prev_grad)]
	}
}

class Pow {

	forward(a, b) {
		b = b.get(0)
		return a.pow(b)
	}

	backward(a, b, prev_grad) {
		b = b.selection.get(0)
		var grad = nj.multiply(a.selection.pow(b-1), b)
		grad = nj.multiply(grad, prev_grad)
		return [grad]
	}
}


// ----- Unary Operations -----

class Max {

	forward(a) {
		return nj.max(a)
	}

	backward(a, prev_grad) {
		var maxVal = nj.max(a.selection)
		var grad = _iterator(a.selection, (x, m, g) => ( (x == m) * g), maxVal, prev_grad.get(0))
		return [grad]
	}
}

class Min {

	forward(a) {
		return nj.min(a)
	}

	backward(a, prev_grad) {
		var minVal = nj.min(a.selection)
		var grad = _iterator(a.selection, (x, m, g) => ( (x == m) * g), minVal, prev_grad.get(0))
		return [grad]
	}
}

class Sum {

	forward(a) {
		return nj.sum(a)
	}

	backward(a, prev_grad) {
		var grad = nj.ones(a.selection.shape)
		grad.assign(prev_grad.get(0), false)
		return [grad]
	}
}

class Exp {

	forward(a) {
		return nj.exp(a)
	}

	backward(a, prev_grad) {
		return a.matmul(prev_grad)
	}
}

class Negative {

	forward(a) {
		return nj.negative(a)
	}

	backward(a, prev_grad) {
		return nj.negative(prev_grad)
	}
}

class Log {

	forward(a) {
		return nj.divide(nj.log(a), 2.303) // convert ln to log (Nerst equation)
	}

	backward(a, prev_grad) {
		var grad = a.pow(-1)
		return nj.multiply(grad, prev_grad)
	}
}

class Transpose {

	forward(a) {
		return a.T
	}

	backward(a, prev_grad) {
		return [a.selection.T]
	}
}

class ReLU {

	forward(a) {
		return _iterator(a, (a) => ((a > 0) * a))
	}

	backward(a, prev_grad) {
		var input = a.selection
		var grad = _iterator(input, (x, g) => ( (x >= 0) * g), prev_grad.get(0))
		return [grad]
	}
}

class ReLU6 {

	forward(a) {
		return _iterator(a, (a) => (Math.min( ...[(a > 0) * a, 6] )))
	}

	backward(a, prev_grad) {
		var input = a.selection
		var grad = _iterator(input, (x, g) => ( (x >= 0) * g), prev_grad.get(0))
		return [grad]
	}
}

class LeakyReLU {

	forward(a) {
		return _iterator(a, (a) => ( ((a > 0) ? 1 : 0.01) * a))
	}

	backward(a, prev_grad) {
		var input = a.selection
		var grad = _iterator(input, (x, g) => ( ((x >= 0) ? 1 : 0.01) * g), prev_grad.get(0))
		return [grad]
	}
}

class Sigmoid {

	forward(a) {
		return nj.sigmoid(a)
	}

	backward(a, prev_grad) {
		var temp = nj.add( nj.negative(nj.sigmoid(a.selection)), 1 )
		var grad = nj.multiply( nj.sigmoid(a.selection), temp )
		grad = nj.multiply(grad, prev_grad)
		return [grad]
	}
}

class Tanh {

	forward(a) {
		return nj.tanh(a)
	}

	backward(a, prev_grad) {
		var temp = nj.tanh(a.selection).pow(2)
		var grad = nj.add(nj.negative(temp), 1)
		grad = nj.multiply(grad, prev_grad)
		return [grad]
	}
}

function _iterator(x, fn, ...args) {
    var out = x.flatten().tolist()

    for (let i = 0; i < out.length; i++) {
    	// + 0 removes negative sign from 0
    	out[i]= fn(out[i], ...args) + 0
    }
    return nj.array(out).reshape(x.shape)
}


module.exports = {
	MatMul,
	Add,
	Sub,
	Pow,
	Max,
	Min,
	Sum,
	Exp,
	Negative,
	Log,
	Transpose,
	ReLU,
	ReLU6,
	LeakyReLU,
	Sigmoid,
	Tanh
}
