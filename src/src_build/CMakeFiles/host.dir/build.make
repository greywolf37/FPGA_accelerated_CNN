# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rubin3737/ml/hw5/project-greywolf37/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rubin3737/ml/hw5/project-greywolf37/src/src_build

# Include any dependencies generated for this target.
include CMakeFiles/host.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/host.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/host.dir/flags.make

CMakeFiles/host.dir/host.cpp.o: CMakeFiles/host.dir/flags.make
CMakeFiles/host.dir/host.cpp.o: ../host.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rubin3737/ml/hw5/project-greywolf37/src/src_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/host.dir/host.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/host.dir/host.cpp.o -c /home/rubin3737/ml/hw5/project-greywolf37/src/host.cpp

CMakeFiles/host.dir/host.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/host.dir/host.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rubin3737/ml/hw5/project-greywolf37/src/host.cpp > CMakeFiles/host.dir/host.cpp.i

CMakeFiles/host.dir/host.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/host.dir/host.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rubin3737/ml/hw5/project-greywolf37/src/host.cpp -o CMakeFiles/host.dir/host.cpp.s

# Object files for target host
host_OBJECTS = \
"CMakeFiles/host.dir/host.cpp.o"

# External object files for target host
host_EXTERNAL_OBJECTS =

host: CMakeFiles/host.dir/host.cpp.o
host: CMakeFiles/host.dir/build.make
host: /home/rubin3737/libtorch/lib/libtorch.so
host: /home/rubin3737/libtorch/lib/libc10.so
host: /home/rubin3737/libtorch/lib/libc10.so
host: CMakeFiles/host.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rubin3737/ml/hw5/project-greywolf37/src/src_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable host"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/host.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/host.dir/build: host

.PHONY : CMakeFiles/host.dir/build

CMakeFiles/host.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/host.dir/cmake_clean.cmake
.PHONY : CMakeFiles/host.dir/clean

CMakeFiles/host.dir/depend:
	cd /home/rubin3737/ml/hw5/project-greywolf37/src/src_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rubin3737/ml/hw5/project-greywolf37/src /home/rubin3737/ml/hw5/project-greywolf37/src /home/rubin3737/ml/hw5/project-greywolf37/src/src_build /home/rubin3737/ml/hw5/project-greywolf37/src/src_build /home/rubin3737/ml/hw5/project-greywolf37/src/src_build/CMakeFiles/host.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/host.dir/depend

