# See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)

project(PetaVision)

################################################################################
# Default values
################################################################################

# Default PetaVision core library directory
set(PV_DIR_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/pv-core")

# Set CMAKE_MODULE_PATH
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(PVConfigProject)
pv_config_project()

################################################################################
# Add subdirectories
################################################################################

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
set(SUBDIR_TXT "${CMAKE_CURRENT_BINARY_DIR}/subdirectories.txt")
if (EXISTS ${SUBDIR_TXT})
    set(PV_SUBDIR_LIST ${SUBDIR_TXT})
    file(READ "${PV_SUBDIR_LIST}" contents)
    string(REGEX REPLACE "\n" ";" contents "${contents}")
    
    foreach(LIN ${contents})
        string(FIND "${LIN}" "#" NUMBERSIGNLOC)
        if (NOT (${NUMBERSIGNLOC} LESS 0) )
            string(SUBSTRING "${LIN}" 0 ${NUMBERSIGNLOC} LIN)
        endif (NOT (${NUMBERSIGNLOC} LESS 0) )
        string(STRIP "${LIN}" LIN)
        if (NOT ("${LIN}" STREQUAL "") )
            message(STATUS "Adding subdirectory \"${LIN}\"")
            add_subdirectory(${LIN} ${LIN})
        endif (NOT ("${LIN}" STREQUAL "") )
    endforeach(LIN ${contents})
    
    # Create a text file called ${SUBDIR_TXT} in the same directory as
    # this CMakeLists.txt file.  That file should contain a list of
    # the projects to include.
    # ${SUBDIR_TXT} is in the .gitignore file so that it won't be
    # (and shouldn't be) committed to the repository.
    # The sample-${SUBDIR_TXT} file illustrates the format.
    # Note that every time you modify ${SUBDIR_TXT} you will need to
    # rerun cmake.
else (EXISTS ${SUBDIR_TXT})
   message("Only building the core PV library.  Create a ${SUBDIR_TXT} file to build additional projects linking to it.")
endif (EXISTS ${SUBDIR_TXT})
