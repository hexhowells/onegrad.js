const fs = require('fs');
var onegrad = require('./../onegrad/tensor')
var utils = require('./../onegrad/utils')
var nj = require("numjs")


function loadTextFile(filepath) {
    return fs.readFileSync(filepath, {encoding:'utf8', flag:'r'}).replace(/[\r\n]/g, "\n");
}


function generateMappings(data) {
    var chars = new Set(data)

    var charsToIdx = {}
    var idxToChars = {}
    for (const [idx, elm] of Array.from(chars).entries()){
        charsToIdx[elm] = idx
        idxToChars[idx] = elm
    }
    return [charsToIdx, idxToChars]
}


class Loader {
    constructor(filepath, seqLength=32) {
        this.data = loadTextFile(filepath)
        this.dataLength = this.data.length

        var [cTi, iTc] = generateMappings(this.data)
        this.charsToIdx = cTi
        this.idxToChars = iTc
        this.vocabSize = Object.keys(this.charsToIdx).length

        this.seqLength = seqLength
        this.pointer = 0

        this.encoder = new utils.OnehotEncoder(this.vocabSize, this.charsToIdx)
    }

    load() {
        var sampleData = this.data.slice(this.pointer, this.pointer+this.seqLength)
        this.pointer += this.seqLength
        if (this.pointer+this.seqLength > this.dataLength) {
            this.pointer = 0
        }
        return this._onehotEncode(sampleData)
    }

    _onehotEncode(data) {
        var encodedData = []
        for (var char of data) {
            var zeros = this.encoder.encode(char)
            encodedData.push(zeros)
        }
        return encodedData
    }
}


module.exports = {Loader}
