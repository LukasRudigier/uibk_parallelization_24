# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cb76/cb761153/uibk_parallelization_24/code_students

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release

# Include any dependencies generated for this target.
include src/IO/CMakeFiles/IO.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/IO/CMakeFiles/IO.dir/compiler_depend.make

# Include the progress variables for this target.
include src/IO/CMakeFiles/IO.dir/progress.make

# Include the compile flags for this target's objects.
include src/IO/CMakeFiles/IO.dir/flags.make

src/IO/CMakeFiles/IO.dir/data_storage.cpp.o: src/IO/CMakeFiles/IO.dir/flags.make
src/IO/CMakeFiles/IO.dir/data_storage.cpp.o: ../src/IO/data_storage.cpp
src/IO/CMakeFiles/IO.dir/data_storage.cpp.o: src/IO/CMakeFiles/IO.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cb76/cb761153/uibk_parallelization_24/code_students/build_release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/IO/CMakeFiles/IO.dir/data_storage.cpp.o"
	cd /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release/src/IO && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/IO/CMakeFiles/IO.dir/data_storage.cpp.o -MF CMakeFiles/IO.dir/data_storage.cpp.o.d -o CMakeFiles/IO.dir/data_storage.cpp.o -c /home/cb76/cb761153/uibk_parallelization_24/code_students/src/IO/data_storage.cpp

src/IO/CMakeFiles/IO.dir/data_storage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/IO.dir/data_storage.cpp.i"
	cd /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release/src/IO && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cb76/cb761153/uibk_parallelization_24/code_students/src/IO/data_storage.cpp > CMakeFiles/IO.dir/data_storage.cpp.i

src/IO/CMakeFiles/IO.dir/data_storage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/IO.dir/data_storage.cpp.s"
	cd /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release/src/IO && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cb76/cb761153/uibk_parallelization_24/code_students/src/IO/data_storage.cpp -o CMakeFiles/IO.dir/data_storage.cpp.s

# Object files for target IO
IO_OBJECTS = \
"CMakeFiles/IO.dir/data_storage.cpp.o"

# External object files for target IO
IO_EXTERNAL_OBJECTS =

src/IO/libIO.a: src/IO/CMakeFiles/IO.dir/data_storage.cpp.o
src/IO/libIO.a: src/IO/CMakeFiles/IO.dir/build.make
src/IO/libIO.a: src/IO/CMakeFiles/IO.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cb76/cb761153/uibk_parallelization_24/code_students/build_release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libIO.a"
	cd /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release/src/IO && $(CMAKE_COMMAND) -P CMakeFiles/IO.dir/cmake_clean_target.cmake
	cd /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release/src/IO && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/IO.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/IO/CMakeFiles/IO.dir/build: src/IO/libIO.a
.PHONY : src/IO/CMakeFiles/IO.dir/build

src/IO/CMakeFiles/IO.dir/clean:
	cd /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release/src/IO && $(CMAKE_COMMAND) -P CMakeFiles/IO.dir/cmake_clean.cmake
.PHONY : src/IO/CMakeFiles/IO.dir/clean

src/IO/CMakeFiles/IO.dir/depend:
	cd /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cb76/cb761153/uibk_parallelization_24/code_students /home/cb76/cb761153/uibk_parallelization_24/code_students/src/IO /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release/src/IO /home/cb76/cb761153/uibk_parallelization_24/code_students/build_release/src/IO/CMakeFiles/IO.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/IO/CMakeFiles/IO.dir/depend

