var nj = require("numjs")


class Function {
	constructor(...tensors) {
		this.parents = tensors;
		this.saved_tensors = [];
	}

	save_for_backward(...tensors) {
		this.saved_tensors.push(...tensors)
	}
}


// ----- Binary Operations -----

class MatMul extends Function {

	forward(a, b) {
		this.save_for_backward(a, b);
		return nj.dot(a, b)
	}

	backward(a, b, prev_grad) {
		var grad_weight = nj.dot(a.selection.T, prev_grad);
		var grad_input = nj.dot(b.selection, prev_grad.T)

		return [grad_input, grad_weight]
	}
}

class Add extends Function {

	forward(a, b) {
		this.save_for_backward(a, b);
		if (b.shape == 1){
			b = b.get(0)
		}
		
		return nj.add(a, b)
	}

	backward(a, b, prev_grad) {
		return [prev_grad, prev_grad]
	}
}

class Sub extends Function {

	forward(a, b) {
		this.save_for_backward(a, b);

		return nj.subtract(a, b)
	}

	backward(a, b, prev_grad) {
		return [prev_grad, prev_grad]
	}
}


// ----- Unary Operations -----

class Max extends Function {

	forward(a) {
		this.save_for_backward(a);
		return nj.max(a)
	}

	backward(prev_grad) {
		// Not yet implemented
	}
}

class Min extends Function {

	forward(a) {
		this.save_for_backward(a);
		return nj.min(a)
	}

	backward(prev_grad) {
		// Not yet implemented
	}
}

class Sum extends Function {

	forward(a) {
		this.save_for_backward(a);
		return nj.sum(a)
	}

	backward(a, prev_grad) {
		var grad = nj.ones(a.selection.shape)
		grad.assign(prev_grad.get(0), false)
		return [grad]
	}
}

class Exp extends Function {

	forward(a) {
		this.save_for_backward(a);
		return nj.exp(a)
	}

	backward(prev_grad) {
		// Not yet implemented
	}
}

class Negative extends Function {

	forward(a) {
		this.save_for_backward(a);
		return nj.negative(a)
	}

	backward(prev_grad) {
		// Not yet implemented
	}
}

class Log extends Function {

	forward(a) {
		this.save_for_backward(a);
		return nj.log(a)
	}

	backward(prev_grad) {
		// Not yet implemented
	}
}

class Transpose extends Function {

	forward(a) {
		this.save_for_backward(a);
		return a.T
	}

	backward(a, prev_grad) {
		return [a.selection.T]
	}
}

class ReLU extends Function {

	forward(a) {
		this.save_for_backward(a);
		return iterator(a, (a) => ((a > 0) * a))
	}

	backward(a, prev_grad) {
		var input = a.selection
		var grad = dualIterator(input, prev_grad, (x, g) => ( (x >= 0) * g) )
		return [grad]
	}
}

class Sigmoid extends Function {

	forward(a) {
		this.save_for_backward(a);
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
