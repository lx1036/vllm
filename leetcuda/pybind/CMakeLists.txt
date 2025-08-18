

file(GLOB SOURCES "*.cpp")
set(PYTHON_EXECUTABLE "python3")
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print([s for s in sys.path if 'packages' in s][0])"
        OUTPUT_VARIABLE PYTHON_SYS_PATH
)
string(STRIP ${PYTHON_SYS_PATH} PYTHON_SYS_PATH)

if("${PYTHON_SYS_PATH}" STREQUAL "")
    message(FATAL_ERROR "Python path is empty! Please check the python env.")
endif()

message("${PYTHON_SYS_PATH}")

cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)

add_subdirectory(pybind11)

pybind11_add_module(engine
        add.cpp
)



