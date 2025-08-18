


#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(engine, m) {
    m.def("add", &add, "Add two numbers");
}
