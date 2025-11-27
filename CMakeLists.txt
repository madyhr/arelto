cmake_minimum_required(VERSION 3.10)

project(rl2 VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED COMPONENTS PNG) 
find_package(SDL2_ttf REQUIRED) 

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
)

add_executable(game_app src/main.cpp)
target_link_libraries(game_app PRIVATE rl2_core)

pybind11_add_module(rl2_py src/bindings.cpp)
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
    # Set PYTHONPATH so it finds the module, and LD_LIBRARY_PATH so it finds the core library
    COMMAND ${CMAKE_COMMAND} -E env 
        PYTHONPATH=$<TARGET_FILE_DIR:rl2_py> 
        LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rl2_core> 
        ${STUBGEN_EXE} rl2_py --output-dir $<TARGET_FILE_DIR:rl2_py>
    COMMENT "Generating Python interface stubs (.pyi) for rl2_py"
    VERBATIM
)

