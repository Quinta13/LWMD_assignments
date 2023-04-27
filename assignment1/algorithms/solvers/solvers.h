#ifndef SOLVERS_H
#define SOLVERS_H

#include "./../model/model.h"

#include <thread>
#include <numeric>

#define VERSION "openmp"  // "openmp" or "thread"

class CommonNeighborSolver {
    /**
     * @brief Class which counts the number of edges in a graph given its edges
     * 
     * It offers methods to compute triangles in graph sequentially or exploiting parallelism
    */

    public:

        CommonNeighborSolver(std::string data_name, int m, int n, int n_thread);
        int solve();
        int get_elapsed_costrunction_time();
        int get_elapsed_solve_time();
        int get_total_time();
        
    private:

        int n_thread;               // number of threads
        Edges edges;                // graph edges
        AdjacencyMatrix mat;         // graph adjacency matrix
        TimeEvaluation solve_time;  // time for solving 

        int sequential_solve();     // sequential
        int parallel_solve_v1();    // parallel - standard library
        int parallel_solve_v2();    // parallel - openmp

        static void parallel_solve_v1_aux(int thread_id, int skip,int* out, std::vector<std::pair<int,int>> edges_, AdjacencyMatrix mat);
};

#endif
