#include "utils.h"

#include <vector>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

/* --------------- TimeEvaluation --------------- */

void TimeEvaluation::start() {
    /**
     * Saves starting timepoint
    */
    this->t1 = std::chrono::high_resolution_clock::now(); 
}
void TimeEvaluation::end() {
    /**
     * Saves ending timepoint
    */
    this->t2 = std::chrono::high_resolution_clock::now();
}

int TimeEvaluation::elapsed() {
    /**
     * @return elapsed time in microseconds
    */
    auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds> (this->t2 - this->t1);
    return duration_ms.count();
}


/* --------------- JsonInfo --------------- */

JsonInfo::JsonInfo(std::string data_name) {
    /**
     * Constructor saving dataset name
     * @param data_name dataset name in datasets folder
    */
    this->data_name = data_name;
}

void JsonInfo::read(int* nodes, int* edges, int* triangles) {
    /**
     * It reads information about the graph from local .json file
     *  number of nodes, edges and trinalges are returned using side effect
     * @param nodes address for number of nodes variable
     * @param edges address for number of edges variable
     * @param triangles address for number of triangles variable
    */


    // local path
    std::filesystem::path path = fs::path(DATA_DIR) / fs::path(this->data_name) / fs::path(INFO_FILE);
    std::ifstream ifs(path);

    std::cout << "- Reading " << path << "... " << std::endl;


    // open file
    json j;
    ifs >> j;
    
    // return values
    *nodes = j[NODES];
    *edges = j[EDGES];
    *triangles = j[TRIANGLES];
    
}

// InputParser

InputParser::InputParser(std::string data_name, int m) {
    /**
     * Constructor saving dataset name and number of edges
     * @param data_name dataset name
     * @param m number of edges
    */

    this->data_name = data_name;
    this->m = m;
}

std::string InputParser::read() {
    /**
     * Reads the local file from dataset folder 
     *
     * @return file content as a string
     */

    std::string file_contents;
    std::string line;

    // local file
    std::filesystem::path path = fs::path(DATA_DIR) / fs::path(this->data_name) / fs::path(EDGE_FILE);
    std::ifstream myfile(path);

    std::cout << "- Reading " << path << "... " << std::endl;

    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            file_contents += line + '\n';
        }
        myfile.close();
    } else {
        std::cout << "Unable to open file " << path << std::endl;
        std::exit(1);
    }

    return file_contents;
}

std::vector<std::pair<int, int>> InputParser::parse_edges() {
    /**
     * Parse edges file creating couples of integers
     * @return vector of edges as a pair of nodes 
    */

    std::vector<std::pair<int, int>> edges(this->m);

    std::istringstream iss(this->read());
    
    int i=0, x, y;

    while (iss >> x >> y) {
        edges[i++] = (std::make_pair(x, y));
    }

    return edges;

}