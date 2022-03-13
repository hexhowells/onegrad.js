
class OnehotEncoder {
    constructor(vocabSize, mapping) {
        this.vocabSize = vocabSize
        this.mapping = mapping
    }

    encode(char) {
        var charEncoding = new Array(this.vocabSize).fill(0)
        var idx = this.mapping[char]
        charEncoding[idx] = 1
        return charEncoding
    }
}

function _randomChoice(arr, p) {
    var rnd = p.reduce( (a, b) => a + b ) * Math.random();
    var idx = p.findIndex( a => (rnd -= a) < 0 );
    return arr[idx]
}

// replication of numpy.random.choice
function randomChoice(arr, p, count=1) {
    return Array.from(Array(count), _randomChoice.bind(null, arr, p));
}

module.exports = {
    OnehotEncoder,
    randomChoice
    }