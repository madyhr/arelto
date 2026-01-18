cmake_minimum_required(VERSION 3.10)

project(rl2 VERSION 1.0)

if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Using Conda prefix: $ENV{CONDA_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED COMPONENTS PNG) 
find_package(SDL2_ttf REQUIRED) 

find_package(OpenMP REQUIRED)

add_subdirectory(extern/pybind11)

file(GLOB_RECURSE ENGINE_SOURCES "src/*.cpp")
list(FILTER ENGINE_SOURCES EXCLUDE REGEX "src/main.cpp")
list(FILTER ENGINE_SOURCES EXCLUDE REGEX "src/bindings.cpp")

add_library(rl2_core SHARED ${ENGINE_SOURCES})

target_include_directories(rl2_core 
    PUBLIC 
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(rl2_core PUBLIC
  SDL2::SDL2
  SDL2_image::SDL2_image
  SDL2_ttf::SDL2_ttf
  pybind11::embed
  OpenMP::OpenMP_CXX
)

add_executable(game_app src/main.cpp)
target_link_libraries(game_app PRIVATE rl2_core)

pybind11_add_module(rl2_py src/bindings.cpp)
set_target_properties(rl2_py PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/rl"
)
target_link_libraries(rl2_py PRIVATE rl2_core)

file(GLOB_RECURSE ASSETS_TO_COPY "assets/*")

add_custom_command(TARGET rl2_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:rl2_core>/assets"
)

find_program(STUBGEN_EXE pybind11-stubgen REQUIRED)

add_custom_command(
    TARGET rl2_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env 
        PYTHONPATH=$<TARGET_FILE_DIR:rl2_py> 
        LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rl2_core> 
        ${STUBGEN_EXE} rl2_py --output-dir $<TARGET_FILE_DIR:rl2_py>
    COMMENT "Generating Python interface stubs (.pyi) for rl2_py"
    VERBATIM
)

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    add_subdirectory(extern/googletest)
    enable_testing()
    file(GLOB_RECURSE TEST_SOURCES "tests/cpp/*.cpp")

if(TEST_SOURCES)
        add_executable(rl2_tests ${TEST_SOURCES})
        target_include_directories(rl2_tests PRIVATE
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/cpp"
        )

        target_link_libraries(rl2_tests PRIVATE
            rl2_core
            gtest
            gtest_main
        )

        include(GoogleTest)
        gtest_discover_tests(rl2_tests)
    else()
        message(STATUS "No test sources found in tests/cpp/, skipping test target")
    endif()
endif()
