'use strict';

var http = require("http"),
    pathUtils = require("path"),
    express = require("express"),
    app = express(),
    PORT = process.env.PORT || 5000,
    appDir = pathUtils.resolve(__dirname);


var graphData = {}

app.use( express.static(appDir) );

app.get("/", function(req, res) {
    res.sendFile( pathUtils.resolve( appDir, "index.html" ) );
});

app.get("/data", function(req, res) {
    res.json(graphData)
});

function launchServer() {
    http.createServer( app ).listen( PORT, function() {
        console.log( "View graph at http://localhost:" + PORT );
    });
}

function visualise(graph={}) {
    graphData = graph;
    launchServer();
}

module.exports = {visualise}