// read and load and return the shapes and tensors from bin files and meta.txt.
#include <iostream>
#include <string>
#include <io.cuh>
#include <fstream>
#include <regex>
#include <vector>
#include <sstream>
#include <filesystem>
#include <cstdlib>
using namespace std;

template <typename T>
void read_bin(const string& path, vector<T>& out, size_t count) {
    uintmax_t expected = count * sizeof(T);
    uintmax_t actual = filesystem::file_size(path);
    if (actual != expected) {
        cerr << path << ": expected " << expected
             << " bytes, got " << actual << endl;
        exit(1);
    }
    out.resize(count);
    ifstream f(path, ios::binary);
    f.read(reinterpret_cast<char*>(out.data()), expected);
    if (!f) {
        cerr << "failed to read " << path << endl;
        exit(1);
    }
}

Fixture load_fixture(const string& folder){ 

    Fixture f;

    ifstream myfile(folder + "/meta.txt");

    string line;
    regex re(R"(shape=\(([^)]*)\))");
    smatch m;

    vector<string> s = {"q", "k", "v", "expected"};
    int i = 0;
    if(!myfile.is_open()){
        cerr << "Error opening file!" << std::endl;
        exit(1);
    }
    while (getline(myfile, line)) {
        smatch m;
        if (regex_search(line, m, re)) {
            string curr_shape = m[1];

            if(i == 1){
                // this is K shape
                string token;
                istringstream shape_string(curr_shape);
                getline(shape_string, token, ',');
                f.H = stoi(token);
                getline(shape_string, token, ',');
                f.L = stoi(token);
                getline(shape_string, token, ',');
                f.D = stoi(token);
            }

            i++;
        }
    }

    // for bin
    read_bin(folder + "/q.bin", f.q, (size_t)f.H * f.D);
    read_bin(folder + "/k.bin", f.K, (size_t)f.H * f.L * f.D);
    read_bin(folder + "/v.bin", f.V, (size_t)f.H * f.L * f.D);
    read_bin(folder + "/expected.bin", f.expected, (size_t)f.H * f.D);

    return f;
}


#ifdef IO_TEST
int main() {
    Fixture f = load_fixture("fixtures/len_100");
    cout << f.H << " " << f.L << " " << f.D << "\n"
         << f.q.size() << " " << f.K.size() << " " << f.V.size() << " "
         << f.expected.size() << "\n";
}
#endif