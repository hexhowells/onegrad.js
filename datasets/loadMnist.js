const fs = require('fs');
var onegrad = require('./../onegrad/tensor')
var nj = require("numjs")


function loadCsv(filepath) {
    const rawData = fs.readFileSync(filepath, {encoding:'utf8', flag:'r'});
    
    var data = []
    for (x of rawData.split('\n')){
        data.push( x.split(',').map(Number) );    
    }
    if (data[data.length-1][0] == ''){
        data.pop()
    }
    
    return data
}


function onehotEncode(values, numClasses=10) {
    if (Array.isArray(values) == false) {
        values = [values]
    }
    var batchSize = values.length
    var encodedTensor = new onegrad.zeros([1]);
    var njEncodedArray = nj.zeros([batchSize, numClasses])

    for (let i=0; i<batchSize; i++) {
        njEncodedArray.set(i, values[i], 1)
    }
    encodedTensor.selection = njEncodedArray
    return encodedTensor
}


function parseDataBatches(rows, batchSize) {
    // temp storage
    var trainingBatch = []
    var labelBatch = []
    // return variables
    var training = []
    var labels = []

    for (row of rows) {
        labelBatch.push(row[0])

        row.shift()
        trainingBatch.push(row)
        if (trainingBatch.length == batchSize) {
            var image = new onegrad.tensor(trainingBatch)
            image.selection = nj.divide(image.selection, 255) // normalisation
            training.push(image)
            
            labels.push(onehotEncode(labelBatch))

            trainingBatch = []
            labelBatch = []
        }
        
    }
    return {data: training, labels: labels}
}


function loadMnist(batchSize=1, loadSample=false) {
    console.log("loading data...")
    if (loadSample) {
        var training = parseDataBatches(loadCsv('../datasets/mnist/mnist_train_100.csv'), batchSize)
        var testing = parseDataBatches(loadCsv('../datasets/mnist/mnist_test_10.csv'), batchSize)
    } else {
        var training = parseDataBatches(loadCsv('../datasets/mnist/mnist_train.csv'), batchSize)
        var testing = parseDataBatches(loadCsv('../datasets/mnist/mnist_test.csv'), batchSize)
    }
    console.log("data loaded!")

    return [training, testing]
}


module.exports = {loadMnist}
