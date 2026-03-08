#include <fstream>
#include <iostream>
#include <string>

#include "rabitqlib/index/ivf/ivf.hpp"

using index_type = rabitqlib::ivf::IVF;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4>\n"
                  << "arg1: path for input IVF index\n"
                  << "arg2: path for output IVF index\n"
                  << "arg3: target format (v2)\n"
                  << "arg4: overwrite output if exists (true/false), false by default\n";
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    std::string target = argv[3];
    bool overwrite = false;
    if (argc > 4 && std::string(argv[4]) == "true") {
        overwrite = true;
    }

    if (target != "v2") {
        std::cerr << "Unsupported target format: " << target << "\n";
        return 1;
    }

    if (!overwrite) {
        std::ifstream probe(output_file, std::ios::binary);
        if (probe.good()) {
            std::cerr << "Output file exists. Pass arg4=true to overwrite.\n";
            return 1;
        }
    }

    index_type ivf;
    ivf.load(input_file);  // auto-detect: legacy or v2
    ivf.save_as_v2(output_file);

    std::cout << "Converted index to v2 format: " << output_file << "\n";
    return 0;
}
