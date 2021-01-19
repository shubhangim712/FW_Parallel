#include <unordered_set>
#include <vector>
#include <algorithm>
#include <iostream>
#include <utility>
#include "ECLgraph.h"
#include "GenerateInput.h"
#include "FW_serial_omp.h"
#include "FW_cuda.h"

static int comp(const float* AdjMat1, const float* AdjMat2, const int nodes) {
  int diffcount = 0;
  for (int i = 0; i < nodes * nodes; ++i) {
    if (AdjMat1[i] != AdjMat2[i]) ++diffcount;
  }
  return diffcount;
}

/*static bool negcycle(const float* AdjMat, const int nodes) {
  for (int i = 0; i < nodes * nodes; i += nodes + 1) {
    if (AdjMat[i] != 0) return true;
  }
  return false;
}*/

int main(int argc, char* argv[]) {
  if (argc != 2) {fprintf(stderr, "USAGE: %s input_graph_name\n\n", argv[0]);  exit(-1);}

  ECLgraph g = generateInput(argv[1]);

  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      if (g.eweight[j] < 0) g.eweight[j] = -g.eweight[j];
    }
  }

  std::cout << "#nodes: " << g.nodes << '\n';
  std::cout << "#edges: " << g.edges << "\n\n";

  float* AdjMat1 = new float [g.nodes * g.nodes];
  FW_cuda(g, AdjMat1);
  std::cout << '\n';

  float* AdjMat2 = new float [g.nodes * g.nodes];
  FW_serial_omp(g, AdjMat2);
  std::cout << '\n';

  const int diffcount = comp(AdjMat1, AdjMat2, g.nodes);
  if (diffcount > 0) std::cout << "Outputs differ by " << diffcount << " values\n";
  else std::cout << "Outputs are equal\n";
  std::cout << '\n';

  delete [] AdjMat1;
  delete [] AdjMat2;
  freeECLgraph(g);
  return 0;
}
