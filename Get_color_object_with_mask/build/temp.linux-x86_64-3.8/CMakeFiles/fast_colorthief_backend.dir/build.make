# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/haobk/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/haobk/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/build/temp.linux-x86_64-3.8

# Include any dependencies generated for this target.
include CMakeFiles/fast_colorthief_backend.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fast_colorthief_backend.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fast_colorthief_backend.dir/flags.make

CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.o: CMakeFiles/fast_colorthief_backend.dir/flags.make
CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.o: ../../cpp/fast_colorthief_backend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/build/temp.linux-x86_64-3.8/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.o -c /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/cpp/fast_colorthief_backend.cpp

CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/cpp/fast_colorthief_backend.cpp > CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.i

CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/cpp/fast_colorthief_backend.cpp -o CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.s

# Object files for target fast_colorthief_backend
fast_colorthief_backend_OBJECTS = \
"CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.o"

# External object files for target fast_colorthief_backend
fast_colorthief_backend_EXTERNAL_OBJECTS =

../lib.linux-x86_64-3.8/fast_colorthief_backend.cpython-38-x86_64-linux-gnu.so: CMakeFiles/fast_colorthief_backend.dir/cpp/fast_colorthief_backend.cpp.o
../lib.linux-x86_64-3.8/fast_colorthief_backend.cpython-38-x86_64-linux-gnu.so: CMakeFiles/fast_colorthief_backend.dir/build.make
../lib.linux-x86_64-3.8/fast_colorthief_backend.cpython-38-x86_64-linux-gnu.so: CMakeFiles/fast_colorthief_backend.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/build/temp.linux-x86_64-3.8/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module ../lib.linux-x86_64-3.8/fast_colorthief_backend.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fast_colorthief_backend.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/build/lib.linux-x86_64-3.8/fast_colorthief_backend.cpython-38-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/fast_colorthief_backend.dir/build: ../lib.linux-x86_64-3.8/fast_colorthief_backend.cpython-38-x86_64-linux-gnu.so

.PHONY : CMakeFiles/fast_colorthief_backend.dir/build

CMakeFiles/fast_colorthief_backend.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fast_colorthief_backend.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fast_colorthief_backend.dir/clean

CMakeFiles/fast_colorthief_backend.dir/depend:
	cd /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/build/temp.linux-x86_64-3.8 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/build/temp.linux-x86_64-3.8 /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/build/temp.linux-x86_64-3.8 /home/haobk/Desktop/code/Segmentation_part_body/visualize/Get_color_object_with_mask/build/temp.linux-x86_64-3.8/CMakeFiles/fast_colorthief_backend.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fast_colorthief_backend.dir/depend

