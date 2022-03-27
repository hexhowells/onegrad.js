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
    nodes.add({id: node, label: `Tensor ${node}`});
  }

  for (var edge of graph.edges) {
    edges.add({from: edge.from, to: edge.to, label: edge.op, arrows: 'to'});
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
        enabled: false
      },
      layout: {
        hierarchical: {
          direction: "UD",
          sortMethod: 'directed'
        }
      },
      nodes: {
        fixed: {
          y: true,
          x: true
        }
      }
    };
    var network = new vis.Network(container, data, options);
}
