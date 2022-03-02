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

	backward(prev_grad) {
		// Not yet implemented
	}
}

class Min {

	forward(a) {
		return nj.min(a)
	}

	backward(prev_grad) {
		// Not yet implemented
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

	backward(prev_grad) {
		// Not yet implemented
	}
}

class Negative {

	forward(a) {
		return nj.negative(a)
	}

	backward(prev_grad) {
		// Not yet implemented
	}
}

class Log {

	forward(a) {
		return nj.log(a)
	}

	backward(prev_grad) {
		// Not yet implemented
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
		return iterator(a, (a) => ((a > 0) * a))
	}

	backward(a, prev_grad) {
		var input = a.selection
		var grad = dualIterator(input, prev_grad, (x, g) => ( (x >= 0) * g) )
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
		grad = nj.dot(grad, prev_grad)
		return [grad]
	}
}

function iterator(x, fn) {
    let out = x.slice().tolist()

    for (let i = 0; i < out.length; i++) {
        for (let j = 0; j < out[i].length; j++) {
        	var tmp = fn(out[i][j])
        	tmp += 0 // removes negative sign from 0
            out[i][j] = tmp
        }
    }
    return nj.array(out)
}

function dualIterator(x, g, fn) {
    let out = x.slice().tolist()
    let grad = g.slice().tolist()

    for (let i = 0; i < out.length; i++) {
        for (let j = 0; j < out[i].length; j++) {
      
        	var tmp = fn(out[i][j], grad[i])
        	tmp += 0 // removes negative sign from 0
            out[i][j] = tmp
        }
    }
    return nj.array(out)
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
