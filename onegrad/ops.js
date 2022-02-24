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


class MatMul extends Function {

	forward(a, b) {
		this.save_for_backward(a, b);
		return nj.dot(a, b)
	}

	backward(prev_grad) {
		var a = this.saved_tensors[0];
		var b = this.saved_tensors[1];
		grad_input = nj.dot(prev_grad, a);
		grad_weight = nj.dot(prev_grad, w)

		return grad_input, grad_weight
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

	backward(prev_grad) {
		// Not yet implemented
	}
}


module.exports = {
	MatMul,
	Add,
	Sub,
	Max,
	Min,
	Sum
}
