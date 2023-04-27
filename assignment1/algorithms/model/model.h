#ifndef MODEL_H
#define MODEL_H

#include "./../utils/utils.h"

#include <vector>
#include <string>
#include <bitset>

/* Classes definition */
class AdjacencyMatrix;
class Edges;

/**
 * @brief Class representing a collection of edges of an undirected graph as pair of nodes, identified as integers
 *      
 * In particular this class provides two methods for:
 *  - read edges from file, provided dataset name
 *  - produce an adjacency matrix
 */
class Edges {

    public:
    
        Edges(std::string data_name, int m);
        int get_m();
        void print_edges();
        std::vector<std::pair<int, int>> get_edges();
        AdjacencyMatrix get_adjacency_matrix(int n, int n_thread);
        
    private:

        int m;                                   // number of edges
        std::vector<std::pair<int, int>> edges;  // edges vector as pair of integers

        static std::string read_edges(std::string data_name);

};

class AdjacencyMatrix {
    /**
     * @brief This class represent a square adjacency Matrix
     * 
     * It has a method to compute the number of common neighbors whithin two nodes, which is used for triangles count.
    */

    public:
        
        AdjacencyMatrix(Edges edges, int n, int n_thread);
        
        int get_n();
        void print_matrix();
        void common_neighbors(int a, int b, int* ret);
        int get_construction_time();

    private:

        int n;                                  // number of nodes (side of the matrix)
        TimeEvaluation construction_time;       // time for matrix construction
        std::vector<std::vector<bool>> matrix;  // n*n adjacency matrix
};

#endif