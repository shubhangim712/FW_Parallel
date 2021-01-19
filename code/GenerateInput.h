#pragma once

#include <vector>
#include "ECLgraph.h"

static void insert(std::vector<std::vector<std::pair<int, int>>>& graph, const int src, const int dst, const int wgt)
{
  const int max = std::max(src, dst);
  if (graph.size() <= max) graph.resize(max + 1);
  std::pair <int, int> edge(dst, wgt);
  graph[src].push_back(edge);
}

static ECLgraph generateInput(const char* const graphInput)
{
  // build graph
  std::vector<std::vector<std::pair<int, int>>> graph;

  if (atoi(graphInput) == 1) {
    insert(graph, 0, 2, -2);
    insert(graph, 1, 0, 4);
    insert(graph, 1, 2, 3);
    insert(graph, 2, 3, 2);
    insert(graph, 3, 1, -1);
  }
  else if (atoi(graphInput) == 2) {
    insert(graph, 0, 1, 5);
    insert(graph, 0, 3, 10);
    insert(graph, 1, 2, 3);
    insert(graph, 2, 3, 1);
  }
  else if (atoi(graphInput) == 3) {
    insert(graph, 0, 1, 3);
    insert(graph, 0, 3, 5);
    insert(graph, 1, 0, 2);
    insert(graph, 1, 3, 4);
    insert(graph, 2, 1, 1);
    insert(graph, 3, 2, 2);
  }
  else if (atoi(graphInput) == 4) {
    insert(graph, 0, 1, 1);
    insert(graph, 1, 2, -1);
    insert(graph, 2, 3, -1);
    insert(graph, 3, 0, -1);
  }
  else if (atoi(graphInput) == 5) {
    insert(graph, 0, 1, 3);
    insert(graph, 0, 2, 8);
    insert(graph, 0, 4, -4);
    insert(graph, 1, 3, 1);
    insert(graph, 1, 4, 7);
    insert(graph, 2, 1, 4);
    insert(graph, 2, 3, -5);
    insert(graph, 3, 0, 2);
    insert(graph, 4, 3, 6);
  }
  else {
    return readECLgraph(graphInput);
  }

  // convert to CSR form
  int edges = 0;
  for (auto s : graph) {
    edges += s.size();
  }

  ECLgraph g;
  g.nodes = graph.size();
  g.edges = edges;
  g.nindex = new int[g.nodes + 1];
  g.nlist = new int[g.edges];
  g.eweight = new int[g.edges];
  int ecnt = 0;
  g.nindex[0] = 0;
  for (int i = 0; i < g.nodes; i++) {
    for (std::pair<unsigned int, int> v : graph[i]) {
      g.nlist[ecnt] = v.first;
      g.eweight[ecnt] = v.second;
      ecnt++;
    }
    g.nindex[i + 1] = ecnt;
  }

  return g;
}
