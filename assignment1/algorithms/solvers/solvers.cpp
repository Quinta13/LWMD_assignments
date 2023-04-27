#include "solvers.h"

#include <numeric>
#include <cmath>
#include <omp.h>

/* --------------- CommonNeighborSolver --------------- */

CommonNeighborSolver::CommonNeighborSolver(std::string data_name, int m, int n, int n_thread):\
    edges(data_name, m), mat(edges.get_adjacency_matrix(n, n_thread)) {
    /**
     * It usese constructor chaining for construct Edges and AdjacencyMatrix
     * @param data_name dataset name
     * @param m number of edges
     * @param n number of nodes
     * @param n_thread number of threads to compute solution with
    */

    this->n_thread = n_thread;
    this->solve_time = TimeEvaluation();
}

int CommonNeighborSolver::get_elapsed_solve_time() {
    /**
     * @return elapsed time for solve evaluation
    */
    return this->solve_time.elapsed();
}

int CommonNeighborSolver::get_elapsed_costrunction_time() {
    /**
     * @return elapsed time for adjacency matrix construction
    */
    return this->mat.get_construction_time();
}

int CommonNeighborSolver::get_total_time() {
    /**
     * @return total elapsed time
    */
    return this->get_elapsed_solve_time() + this->get_elapsed_costrunction_time();
}

int CommonNeighborSolver::solve() {
    /**
     * It solve triangle count problem running the solver
     *  it run sequqntial in case of one thread, parallael otherwise
     * @return number of triangles in the graph
    */

    int n_triangles; 

    std::cout << "- Solving... " << std::endl;

    this->solve_time.start();

    if (this->n_thread == 1) {
        n_triangles =  this->sequential_solve();
    } else {
        if(VERSION == "thread") {
            n_triangles =  this->parallel_solve_v1();  // choose version 1
        } else if(VERSION == "openmp") {
            n_triangles =  this->parallel_solve_v2();  // choose version 2
        } else {
            std::cout << "Invalid version " << VERSION << std::endl;
            std::exit(1);
        }
    
    }

    this->solve_time.end();

    std::cout << "- Complete " << std::endl << std::endl;

    return n_triangles;

}

int CommonNeighborSolver::sequential_solve() {
    /**
     * Perform sequential execution
     * @return number of triangles in the graph
    */

    int triangles = 0;

    auto edges_ = this->edges.get_edges();

    for(int i=0; i < edges_.size(); i++) {
        auto edge = edges_[i];
        this->mat.common_neighbors(edge.first, edge.second, &triangles);
    }

    triangles /= 3;

    return triangles;

}

int CommonNeighborSolver::parallel_solve_v1() {
    /**
     * Perform parallel execution with standard library
     * @return triangles in the graph
    */

    std::vector<int> triangles(this->n_thread, 0);
    std::vector<std::thread> threads(this->n_thread); // vectors for threads

    std::cout << "- Starting threads... " << std::endl;
    for (int i = 0; i < this->n_thread; i++) {
        threads[i] = std::thread(
            this->parallel_solve_v1_aux, 
            i,
            this->n_thread, 
            &triangles[i], // return output
            this->edges.get_edges(),
            this->mat
        );
    }

    // waiting to comlpete
    std::cout << "- Waiting for threads to complete... " << std::endl;
    for (int i = 0; i < this->n_thread; i++) {
        threads[i].join();
    }

    std::cout << "- Combining results... " << std::endl;
    int sum = std::accumulate(triangles.begin(), triangles.end(), 0);
    sum /= 3;

    return sum;

}

int CommonNeighborSolver::parallel_solve_v2() {
    /**
     * Perform parallel execution using openmp
     * @return triangles in the graph
    */

    int triangles, x=0;

    auto edges_ = this->edges.get_edges();

    std::cout << "- Starting threads... " << std::endl;
    #pragma omp parallel for \
    num_threads(this->n_thread) \
    reduction(+:triangles) \
    firstprivate(x)

    for(int i=0; i < edges_.size(); i++) {
        x = 0;
        auto edge = edges_[i];
        this->mat.common_neighbors(edge.first, edge.second, &x);
        triangles += x;
    }

    std::cout << "- Combining results" << std::endl;

    triangles /= 3;

    return triangles;

}

void CommonNeighborSolver::parallel_solve_v1_aux(int thread_id, int skip, int* out, std::vector<std::pair<int,int>> edges_, AdjacencyMatrix mat) {
    /**
     * Auxiliar function to compute number of triangles for a certain thread,
     *  each thread is responsible just of a subset of edges in the collection (with no overlap)
     * @param thread_id id of thread, corresponding with initial index in the collection
     * @param skip skip in the collection, corresponding to total number of threads
     * @param edges_ collection of edges as pair of nodes
     * @param mat adjacency matrix
    */
    
    for(int i=thread_id; i<edges_.size(); i+=skip) {
        auto edge = edges_[i];
        mat.common_neighbors(edge.first, edge.second, out);
    }
}