# Minimum required version for CMake
cmake_minimum_required(VERSION 3.10)

# The name of your project
project(MyCppGame VERSION 1.0)

# Set the C++ standard to use (essential for modern C++)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- CORRECTED PACKAGE FINDING ---
# Find the required SDL2 base library.
# Removed 'COMPONENTS Main' to avoid configuration errors on some systems.
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED COMPONENTS PNG) 
find_package(SDL2_ttf REQUIRED) 

# Add the executable target. This tells CMake to compile main.cpp
add_executable(game_executable src/main.cpp)

# --- RESOURCE COPYING STEP ---
# The source directory of your project is ${CMAKE_SOURCE_DIR}.
# The directory where the executable is built is ${CMAKE_CURRENT_BINARY_DIR}.

# Copy the resource file from the source directory to the build directory.
file(COPY "${CMAKE_SOURCE_DIR}/textures/wizard.png"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
)
file(COPY "${CMAKE_SOURCE_DIR}/textures/goblin.png"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
)
file(COPY "${CMAKE_SOURCE_DIR}/textures/grassy_plains.png"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
)
# Link the SDL libraries to the executable
# Using PRIVATE linkage and the modern imported targets exclusively.
# SDL2::Main is still included in the linking step for proper SDL entry point handling.
target_link_libraries(game_executable PRIVATE
  SDL2::SDL2 # Base SDL library
  SDL2_image::SDL2_image # Image loading
  SDL2_ttf::SDL2_ttf # Font/text loading
)
