# Minimum required version for CMake
cmake_minimum_required(VERSION 3.10)

# The name of your project
project(MyCppGame VERSION 1.0)

# Set the C++ standard to use (essential for modern C++)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the required SDL2 packages
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED) # For loading images
find_package(SDL2_ttf REQUIRED)  # For fonts/text

# Add the executable target. This tells CMake to compile main.cpp
add_executable(game_executable src/main.cpp)

# Link the SDL libraries to the executable
target_link_libraries(game_executable PUBLIC
    SDL2::SDL2
    ${SDL2_IMAGE_LIBRARIES}
    ${SDL2_TTF_LIBRRARIES}
)
