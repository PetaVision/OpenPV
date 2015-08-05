# This CMakeLists.txt file can be used to generate makefiles for several projects at once.
# It should be put in the directory containing the projects.

project(PetaVision)


# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8)


# Set variable for PetaVision directory (the directory that contains
# the core library's source directory)
set(PV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pv-core CACHE PATH "The core PetaVision directory")


#Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${PV_DIR}/docs/cmake/Modules/")


# Adds the core PetaVision directory
add_subdirectory(${PV_DIR})


# Adds other projects that link to PetaVision.  Their CMakeLists.txt
# file should use the cache variable PV_DIR to point to PetaVision.
# Create a text file called subdirectories.txt in the same directory
# as this CMakeLists.txt file.
# 
# Comments are marked with the number sign '#'; it and everything
# following it are ignored.  Leading and trailing whitespace is then
# stripped  If the resulting line is nonempty, its contents are interpreted
# as a directory and and passed to the CMake add_subdirectory command.
if (EXISTS subdirectories.txt)
    set(PV_SUBDIR_LIST subdirectories.txt)
    file(READ "${PV_SUBDIR_LIST}" contents)
    string(REGEX REPLACE "\n" ";" contents "${contents}")
    
    foreach(LIN ${contents})
        string(FIND "${LIN}" "#" NUMBERSIGNLOC)
        if (NOT (${NUMBERSIGNLOC} LESS 0) )
            string(SUBSTRING "${LIN}" 0 ${NUMBERSIGNLOC} LIN)
        endif (NOT (${NUMBERSIGNLOC} LESS 0) )
        string(STRIP "${LIN}" LIN)
        if (NOT ("${LIN}" STREQUAL "") )
            message("Adding subdirectory \"${LIN}\"")
            add_subdirectory(${LIN} ${LIN})
        endif (NOT ("${LIN}" STREQUAL "") )
    endforeach(LIN ${contents})
    
    # Create a text file called subdirectories.txt in the same directory as
    # this CMakeLists.txt file.  That file should contain a list of
    # the projects to include.
    # subdirectories.txt is in the .gitignore file so that it won't be
    # (and shouldn't be) committed to the repository.
    # The sample-subdirectories.txt file illustrates the format.
    # Note that every time you modify subdirectories.txt you will need to
    # rerun cmake.
else (EXISTS subdirectories.txt)
   message("Only building the core PV library.  Create a subdirectories.txt file to build additional projects linking to it.")
endif (EXISTS subdirectories.txt)
