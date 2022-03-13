
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

module.exports = {OnehotEncoder}