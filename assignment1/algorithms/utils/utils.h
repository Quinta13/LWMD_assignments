#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>

#define DATA_DIR "./../data"    // Relative path to datasets folder
#define EDGE_FILE "edges.txt"   // Name of edges file in each specific dataset folder
#define INFO_FILE "info.json"   // Name of information file in each specific dataset folder, 
                                //  containing number of nodes, edges and trees
#define NODES "nodes"         // json key for nodes in info.json
#define EDGES "edges"         // json key for edges in info.json
#define TRIANGLES "triangles" // json key for traingles in info.json

class TimeEvaluation;
class InputParser;

/**
 * @brief Util class to compute elapsded evaluation in time
 *
 * This class provides some methods to evaluate execution time of a certain block of code
 *  - start, to be invoked before the evaluation;
 *  - end,   to be invoked after the evaluation;
 *  - elapsed, provides the elapsed time.
 */
class TimeEvaluation {

    public:

        void start();
        void end();
        int elapsed();

    private:
    
        std::chrono::time_point<std::chrono::high_resolution_clock> t1;  // initial time-point
        std::chrono::time_point<std::chrono::high_resolution_clock> t2;  // final   time-point
        
};

class JsonInfo {
    /**
     * @brief Util class to read local .json file
     * It provides a method to read graph information of a certain datasets
    */
    public:

        JsonInfo(std::string data_name);
        void read(int* nodes, int* edges, int* triangles);

    private:

        std::string data_name;  // dataset name
};

class InputParser {
    /**
     * @brief Util class to parse edges.txt
     * It provides some methods to parse the file containing a couple of integers per line
    */

    public:

        InputParser(std::string data_name, int m);
        std::vector<std::pair<int, int>> parse_edges();

    private:

        int m;                  // number of edges
        std::string data_name;  // dataset name

        std::string read();
};


#endif
