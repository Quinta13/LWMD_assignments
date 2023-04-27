// g++ main.cpp model/model.cpp utils/utils.cpp solvers/solvers.cpp -o main -O3 -std=c++17 -fopenmp -lpthread

#include <iostream>
#include <vector>
#include "./solvers/solvers.h"

int main(int argc, char **argv) {

    // Argument parsing

    if (argc != 3) {
        std::cout << " Usage: " << argv[0] << " <data_name> <num_threads> " << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const std::string DATA_NAME = argv[1]; // Dataset name
    const int N_THREAD = atoi(argv[2]);    // Number of threads

    JsonInfo info(DATA_NAME);

    int N, M, T;

    std::cout << std::endl;
    info.read(&N, &M, &T);

    std::cout << std::endl;
    std::cout << "Dataset: " << DATA_NAME << std::endl;
    std::cout << " > Number of nodes:     " << N        << std::endl;
    std::cout << " > Number of edges:     " << M        << std::endl;
    std::cout << " > Number of triangles: " << T        << std::endl;
    std::cout << " > Number of threads:   " << N_THREAD << std::endl;
    std::cout << std::endl;

    CommonNeighborSolver solver(DATA_NAME, M, N, N_THREAD);

    int n_triangles; 
    
    n_triangles = solver.solve();

    if (n_triangles == T) {
        std::cout << "Correct - Execution time: " <<\
            solver.get_elapsed_solve_time() / 1000000. << " sec" << std::endl << std::endl;
    } else {
        std::cout << "Wrong: " << n_triangles << " triangles found " << std::endl;
    }

    return 0;
}