# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/geniusdegenerate/aBunchOfSlam

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/geniusdegenerate/aBunchOfSlam/build

# Include any dependencies generated for this target.
include CMakeFiles/icp_svd.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/icp_svd.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/icp_svd.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/icp_svd.dir/flags.make

CMakeFiles/icp_svd.dir/codegen:
.PHONY : CMakeFiles/icp_svd.dir/codegen

CMakeFiles/icp_svd.dir/src/icp_svd.cpp.o: CMakeFiles/icp_svd.dir/flags.make
CMakeFiles/icp_svd.dir/src/icp_svd.cpp.o: /home/geniusdegenerate/aBunchOfSlam/src/icp_svd.cpp
CMakeFiles/icp_svd.dir/src/icp_svd.cpp.o: CMakeFiles/icp_svd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/geniusdegenerate/aBunchOfSlam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/icp_svd.dir/src/icp_svd.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/icp_svd.dir/src/icp_svd.cpp.o -MF CMakeFiles/icp_svd.dir/src/icp_svd.cpp.o.d -o CMakeFiles/icp_svd.dir/src/icp_svd.cpp.o -c /home/geniusdegenerate/aBunchOfSlam/src/icp_svd.cpp

CMakeFiles/icp_svd.dir/src/icp_svd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/icp_svd.dir/src/icp_svd.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/geniusdegenerate/aBunchOfSlam/src/icp_svd.cpp > CMakeFiles/icp_svd.dir/src/icp_svd.cpp.i

CMakeFiles/icp_svd.dir/src/icp_svd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/icp_svd.dir/src/icp_svd.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/geniusdegenerate/aBunchOfSlam/src/icp_svd.cpp -o CMakeFiles/icp_svd.dir/src/icp_svd.cpp.s

CMakeFiles/icp_svd.dir/src/eigen.cpp.o: CMakeFiles/icp_svd.dir/flags.make
CMakeFiles/icp_svd.dir/src/eigen.cpp.o: /home/geniusdegenerate/aBunchOfSlam/src/eigen.cpp
CMakeFiles/icp_svd.dir/src/eigen.cpp.o: CMakeFiles/icp_svd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/geniusdegenerate/aBunchOfSlam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/icp_svd.dir/src/eigen.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/icp_svd.dir/src/eigen.cpp.o -MF CMakeFiles/icp_svd.dir/src/eigen.cpp.o.d -o CMakeFiles/icp_svd.dir/src/eigen.cpp.o -c /home/geniusdegenerate/aBunchOfSlam/src/eigen.cpp

CMakeFiles/icp_svd.dir/src/eigen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/icp_svd.dir/src/eigen.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/geniusdegenerate/aBunchOfSlam/src/eigen.cpp > CMakeFiles/icp_svd.dir/src/eigen.cpp.i

CMakeFiles/icp_svd.dir/src/eigen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/icp_svd.dir/src/eigen.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/geniusdegenerate/aBunchOfSlam/src/eigen.cpp -o CMakeFiles/icp_svd.dir/src/eigen.cpp.s

CMakeFiles/icp_svd.dir/src/pangolin.cpp.o: CMakeFiles/icp_svd.dir/flags.make
CMakeFiles/icp_svd.dir/src/pangolin.cpp.o: /home/geniusdegenerate/aBunchOfSlam/src/pangolin.cpp
CMakeFiles/icp_svd.dir/src/pangolin.cpp.o: CMakeFiles/icp_svd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/geniusdegenerate/aBunchOfSlam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/icp_svd.dir/src/pangolin.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/icp_svd.dir/src/pangolin.cpp.o -MF CMakeFiles/icp_svd.dir/src/pangolin.cpp.o.d -o CMakeFiles/icp_svd.dir/src/pangolin.cpp.o -c /home/geniusdegenerate/aBunchOfSlam/src/pangolin.cpp

CMakeFiles/icp_svd.dir/src/pangolin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/icp_svd.dir/src/pangolin.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/geniusdegenerate/aBunchOfSlam/src/pangolin.cpp > CMakeFiles/icp_svd.dir/src/pangolin.cpp.i

CMakeFiles/icp_svd.dir/src/pangolin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/icp_svd.dir/src/pangolin.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/geniusdegenerate/aBunchOfSlam/src/pangolin.cpp -o CMakeFiles/icp_svd.dir/src/pangolin.cpp.s

# Object files for target icp_svd
icp_svd_OBJECTS = \
"CMakeFiles/icp_svd.dir/src/icp_svd.cpp.o" \
"CMakeFiles/icp_svd.dir/src/eigen.cpp.o" \
"CMakeFiles/icp_svd.dir/src/pangolin.cpp.o"

# External object files for target icp_svd
icp_svd_EXTERNAL_OBJECTS =

/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: CMakeFiles/icp_svd.dir/src/icp_svd.cpp.o
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: CMakeFiles/icp_svd.dir/src/eigen.cpp.o
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: CMakeFiles/icp_svd.dir/src/pangolin.cpp.o
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: CMakeFiles/icp_svd.dir/build.make
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/lib/x86_64-linux-gnu/libpython3.10.so
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_glgeometry.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_python.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_scene.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_tools.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_video.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_geometry.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libtinyobj.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_plot.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_display.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_vars.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_windowing.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_opengl.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/lib/x86_64-linux-gnu/libEGL.so
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/lib/x86_64-linux-gnu/libOpenGL.so
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/lib/x86_64-linux-gnu/libepoxy.so
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_image.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_packetstream.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: /usr/local/lib/libpango_core.so.0.9.2
/home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd: CMakeFiles/icp_svd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/geniusdegenerate/aBunchOfSlam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable /home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/icp_svd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/icp_svd.dir/build: /home/geniusdegenerate/aBunchOfSlam/scripts/icp_svd
.PHONY : CMakeFiles/icp_svd.dir/build

CMakeFiles/icp_svd.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/icp_svd.dir/cmake_clean.cmake
.PHONY : CMakeFiles/icp_svd.dir/clean

CMakeFiles/icp_svd.dir/depend:
	cd /home/geniusdegenerate/aBunchOfSlam/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/geniusdegenerate/aBunchOfSlam /home/geniusdegenerate/aBunchOfSlam /home/geniusdegenerate/aBunchOfSlam/build /home/geniusdegenerate/aBunchOfSlam/build /home/geniusdegenerate/aBunchOfSlam/build/CMakeFiles/icp_svd.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/icp_svd.dir/depend

