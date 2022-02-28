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
		return nj.add(a, b)
	}

	backward(prev_grad) {
		// Not yet implemented
	}
}

class Sub extends Function {

	forward(a, b) {
		this.save_for_backward(a, b);
		return nj.subtract(a, b)
	}

	backward(prev_grad) {
		// Not yet implemented
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

class ReLU extends Function {

	forward(a) {
		this.save_for_backward(a);
		return iterator(a, a => ((a > 0) * a))
	}

	backward(prev_grad) {
		// Not yet implemented
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
	ReLU
}
