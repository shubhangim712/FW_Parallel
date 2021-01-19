#pragma once
#include "ECLgraph.h"
#include <limits>
#include <sys/time.h>

//utility function
static void printGraph(const float* AdjMat, const int nodes) {
  for (int i = 0; i < nodes; ++i) {
    for (int j = 0; j < nodes; ++j) {
      std::cout << AdjMat[i * nodes + j] << ' ';
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}

static void FW_serial_omp(ECLgraph g, float* AdjMat) {
  timeval start, end;
  gettimeofday(&start, NULL);
  int nodes = g.nodes;

  #pragma omp parallel for default(none) shared(nodes, AdjMat)
  for (int i = 0; i < nodes; ++i) {
    for (int j = 0; j < nodes; ++j) {
      AdjMat[i * nodes + j] = (i == j) ? 0 : std::numeric_limits<float>::infinity();
    }
  }
  #pragma omp parallel for default(none) shared(g, AdjMat)
  for (int i = 0; i < g.nodes; ++i) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; ++j) {
      AdjMat[i * g.nodes + g.nlist[j]] = g.eweight[j];
    }
  }
  
  gettimeofday(&end, NULL);
  const double inittime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("time to initialize (serial/omp): %.5f s\n", inittime);

  if (g.nodes <= 10) printGraph(AdjMat, g.nodes);

  gettimeofday(&start, NULL);	
  
  for (int k = 0; k < nodes; ++k) {
  #pragma omp parallel for default(none) shared(k, nodes, AdjMat)
  for (int i = 0; i < nodes; ++i) {
    for (int j = 0; j < nodes; ++j) {
      if (AdjMat[i * nodes + j] > AdjMat[i * nodes + k] + AdjMat[k * nodes + j])
        AdjMat[i * nodes + j] = AdjMat[i * nodes + k] + AdjMat[k * nodes + j];
      }
    }
   if (g.nodes <= 5) {std::cout << "---k = " << k << "---\n"; printGraph(AdjMat, g.nodes); }
  }
  
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time (serial/omp): %.5f s\n", runtime);

  if (g.nodes <= 10) printGraph(AdjMat, g.nodes);
}
