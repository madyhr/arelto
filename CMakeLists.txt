# Minimum required version for CMake
cmake_minimum_required(VERSION 3.10)

# The name of your project
project(MyCppGame VERSION 1.0)

# Set the C++ standard to use (essential for modern C++)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- FIND EXTERNAL LIBRARIES ---
# Find the required SDL2 base library and its extensions
# Note: SDL2_ttf isn't used in your current code, but we keep it for convention.
find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED COMPONENTS PNG) 
find_package(SDL2_ttf REQUIRED) 

# --- GATHER PROJECT FILES ---
# 1. Collect all implementation files from the src directory
file(GLOB_RECURSE GAME_SOURCES "src/*.cpp")

# 2. Add the executable target using all source files
# This now includes src/main.cpp, src/Game.cpp, src/Player.cpp, src/Enemy.cpp, etc.
add_executable(game.exe ${GAME_SOURCES})

# --- CONFIGURATION FOR COMPILER AND LINKER ---

# 3. Tell the compiler where to find *your* header files.
# The 'include' directory is the public interface for your classes.
# INTERFACE means this is only used for compilation and not linking.
target_include_directories(game.exe 
	PUBLIC 
	"${CMAKE_SOURCE_DIR}/include"
)

# 4. Link the SDL libraries to the executable
# Using PRIVATE linkage and the modern imported targets.
target_link_libraries(game.exe PRIVATE
  SDL2::SDL2              # Base SDL library
  SDL2_image::SDL2_image  # Image loading
  SDL2_ttf::SDL2_ttf      # Font/text loading (kept for completeness)
)

# --- RESOURCE COPYING STEP (Simplified) ---
# Better to copy entire 'assets' folder rather than individual files.
# You'd copy the contents of your 'assets' directory to the build directory.
file(GLOB_RECURSE ASSETS_TO_COPY "assets/*")

add_custom_command(TARGET game.exe POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_SOURCE_DIR}/assets"
		"$<TARGET_FILE_DIR:game.exe>/assets"
)
