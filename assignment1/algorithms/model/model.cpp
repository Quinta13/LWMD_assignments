#include "model.h"


/* --------------- Edges --------------- */

Edges::Edges(std::string data_name, int m) {
    /**
     * Constructor savig number of edges and edges data structure
     *
     * @param data_name dataset name
     * @param m number of edges
     */

    this->m = m;

    InputParser in_parse(data_name, m);

    this->edges = in_parse.parse_edges();
}

std::vector<std::pair<int, int>> Edges::get_edges() {
    /**
     * @return vector of couple of edges
    */

   return this->edges;
}

AdjacencyMatrix Edges::get_adjacency_matrix(int n, int n_thread) {
    /**
     * Return corresponding adjacency matrix
     * @param n number of nodes
     * @param n_thread number of threads for matrix building
     * @return adjacency matrix
    */

   AdjacencyMatrix mat(*this, n, n_thread);
   return mat;
}

void Edges::print_edges() {
    /**
     * Print the edges as a couple of integers
    */

    for(auto edge : this->edges) {
        std::cout << "(" << edge.first << ", " << edge.second << ")"  << std::endl;
    }
}

int Edges::get_m() { 
    /**
     * @return number of edges
    */
    
    return this->m; 
}


/* --------------- adjacencyMatrix --------------- */

AdjacencyMatrix::AdjacencyMatrix(Edges edges, int n, int n_thread) {
    /**
     * Build the adjacency matrix based on the edges.
     * It exploits time evaluation utility
     * @param edges vector of pair of nodes
     * @param n number of nodes
     * @param n_thread number of threads for matrix building
     * @return adjacency matrix
    */

    this->n = n;

    std::vector<std::vector<bool>> matrix(n, std::vector<bool>(n, false));

    this->construction_time.start();

    auto edges_ = edges.get_edges();

    std::cout << "- Creating adjacency matrix... " << std::endl;

    // #pragma omp parallel if(n_thread!=1) num_threads(n_thread) TODO
    for(int i=0; i<edges_.size(); i++) {
        auto edge = edges_[i];
        matrix[edge.first ][edge.second] = true;
        matrix[edge.second][edge.first ] = true;
    }

    this->construction_time.end();

    std::cout << "- Complete - " << this->get_construction_time() / 1000000. << " sec " << std::endl;

    this->matrix = matrix;
}

int AdjacencyMatrix::get_construction_time() {
    /**
     * @return total matrix construction elapsed time
    */

    return this->construction_time.elapsed();
}

void AdjacencyMatrix::print_matrix() {
    /**
     * Print the adjacency matrix.
    */

    for (int i = 0; i < this->n; i++) {
        for (int j = 0; j < this->n; j++) {
            std::cout << this->matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
}

int AdjacencyMatrix::get_n() {
    /**
     * @return number of nodes
    */
    return this->n;
}


void AdjacencyMatrix::common_neighbors(int a, int b, int* ret) {
    /**
     * Count common neighbors between two points
     * Return value is summed up to the given address, in particular variable ret should
     *  - be initialized with 0 in case of single return value
     *  - be a buffer where to sum up partial results
     * @param a index of first node of the edge
     * @param b index of second node of the edge
     * @param ret address for a variable to sum number of common neighbors
    */

   int neighbors = 0;

    for(int i=0; i<this->n; i++) {
        neighbors += this->matrix[a][i] && this->matrix[b][i];
    }

    *ret += neighbors;

}