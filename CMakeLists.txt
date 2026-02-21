cmake_minimum_required(VERSION 3.15)

project(arelto VERSION 1.0)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS On)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(SDL2_ttf REQUIRED)
find_package(SDL2_mixer REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(extern/pybind11)

file(GLOB_RECURSE ENGINE_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
list(FILTER ENGINE_SOURCES EXCLUDE REGEX "src/main.cpp")
list(FILTER ENGINE_SOURCES EXCLUDE REGEX "src/bindings.cpp")

add_library(arelto_core SHARED ${ENGINE_SOURCES})

target_include_directories(arelto_core
    PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(arelto_core PUBLIC
  SDL2::SDL2
  SDL2_image::SDL2_image
  SDL2_ttf::SDL2_ttf
  SDL2_mixer::SDL2_mixer
  pybind11::pybind11
  OpenMP::OpenMP_CXX
)

install(TARGETS arelto_core DESTINATION rl)

add_executable(game_app src/main.cpp)
target_link_libraries(game_app PRIVATE arelto_core SDL2::SDL2main)

pybind11_add_module(arelto_py src/bindings.cpp)
target_link_libraries(arelto_py PRIVATE arelto_core)

add_custom_command(TARGET arelto_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/assets"
        "$<TARGET_FILE_DIR:arelto_core>/assets"
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/assets/" DESTINATION rl/assets)
install(TARGETS arelto_py DESTINATION rl)

option(BUILD_STUBS "Generate Python interface stubs (.pyi)" OFF)

if(BUILD_STUBS)
    find_program(STUBGEN_EXE pybind11-stubgen)
    if(STUBGEN_EXE)
        if(WIN32)
            set(PYTHONPATH_SEP ";")
            set(LIB_PATH_VAR "PATH")
        elseif(APPLE)
            set(PYTHONPATH_SEP ":")
            set(LIB_PATH_VAR "DYLD_LIBRARY_PATH")
        else()
            set(PYTHONPATH_SEP ":")
            set(LIB_PATH_VAR "LD_LIBRARY_PATH")
        endif()

        add_custom_command(
            TARGET arelto_py POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env 
                "PYTHONPATH=$<TARGET_FILE_DIR:arelto_py>${PYTHONPATH_SEP}${CMAKE_SOURCE_DIR}${PYTHONPATH_SEP}$ENV{PYTHONPATH}"
                "${LIB_PATH_VAR}=$<TARGET_FILE_DIR:arelto_py>${PYTHONPATH_SEP}$ENV{${LIB_PATH_VAR}}"
                ${STUBGEN_EXE} -o $<TARGET_FILE_DIR:arelto_py> arelto_py
            COMMENT "Generating Python interface stubs (.pyi)"
            VERBATIM
        )
        install(FILES $<TARGET_FILE_DIR:arelto_py>/arelto_py.pyi DESTINATION rl OPTIONAL)
    else()
        message(WARNING "pybind11-stubgen not found. Skipping stub generation.")
    endif()
endif()

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(extern/googletest)
    enable_testing()
    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/cpp/*.cpp")

    if(TEST_SOURCES)
        add_executable(arelto_tests ${TEST_SOURCES})
        target_include_directories(arelto_tests PRIVATE
            "${CMAKE_SOURCE_DIR}/include"
            "${CMAKE_SOURCE_DIR}/tests/cpp"
        )

        target_link_libraries(arelto_tests PRIVATE
            arelto_core
            gtest
            gtest_main
        )

        include(GoogleTest)
        gtest_discover_tests(arelto_tests)
    else()
        message(STATUS "No test sources found in tests/cpp/, skipping test target")
    endif()
endif()
