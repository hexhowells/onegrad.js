let url = window.location.href + "data";


fetch(url)
.then(res => res.json())
.then(out => {
  const [nodes, edges] = createGraph(out)
  showGraph(nodes, edges)
})
  

function createGraph(graph) {
  var nodes = new vis.DataSet();
  var edges = new vis.DataSet();

  for (var node of graph.nodes) {
    var tensorName = (node.label) ? node.label : `Tensor ${node.id}`
    var nodeColor = (node.label) ? 'rgba(140, 188, 252, 1)' : 'rgba(185, 220, 252, 1)'
    nodes.add({
      id: node.id, 
      label: `<b>${tensorName}</b>\nop: ${node.op}\nshape: ${node.shape}\nrequires grad: ${node.requiresGrad}`, 
      shape:'box',
      color: {background: nodeColor},
    });
  }

  for (var edge of graph.edges) {
    edges.add({from: edge.from, to: edge.to, arrows: 'to'});
  }

  return [nodes, edges]
}


function showGraph(nodes, edges) {
  var container = document.getElementById("mynetwork");
    var data = {
      nodes: nodes,
      edges: edges,
    };
    var options = {
      physics: {
        stabilizations: false,
        enabled: true,
        hierarchicalRepulsion: {
          avoidOverlap: 1,
        }
      },
      layout: {
        hierarchical: {
          direction: "UD",
          sortMethod: 'directed'
        }
      },
      nodes: {
        margin: 10,
        font: {
          align: 'left',
          multi: 'html'
        },
        shapeProperties: {
          interpolation: false    // 'true' for intensive zooming
        }
        /*fixed: {
          y: true,
          x: true
        }*/
      }
    };
    var network = new vis.Network(container, data, options);

    network.on("stabilizationIterationsDone", function () {
      network.setOptions( { physics: false } );
  });
}
