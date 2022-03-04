var nj = require("numjs")


// ----- Binary Operations -----

class MatMul {

	forward(a, b) {
		return nj.dot(a, b)
	}

	backward(a, b, prev_grad) {
		var grad_weight = nj.dot(a.selection.T, prev_grad);
		var grad_input = nj.dot(b.selection, prev_grad.T)

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
		return nj.log(a)
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
	Sigmoid
}
