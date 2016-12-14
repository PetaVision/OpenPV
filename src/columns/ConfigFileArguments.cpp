/*
 * ConfigFileArguments.cpp
 *
 *  Created on: Sep 21, 2015
 *      Author: pschultz
 */

#include "ConfigFileArguments.hpp"
#include "include/pv_common.h"
#include "utils/PVLog.hpp"
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>

namespace PV {

ConfigFileArguments::ConfigFileArguments(
      std::string const &configFile,
      MPI_Comm communicator,
      bool allowUnrecognizedArguments) {
   initialize_base();
   initialize(configFile, communicator, allowUnrecognizedArguments);
}

int ConfigFileArguments::initialize_base() { return PV_SUCCESS; }

int ConfigFileArguments::initialize(
      std::string const &configFile,
      MPI_Comm communicator,
      bool allowUnrecognizedArguments) {
   resetState(configFile, communicator, allowUnrecognizedArguments);
   return PV_SUCCESS;
}

void ConfigFileArguments::resetState(
      std::string const &configFile,
      MPI_Comm communicator,
      bool allowUnrecognizedArguments) {
   std::string configContents;
   unsigned int fileSize = 0U;
   int rank;
   MPI_Comm_rank(communicator, &rank);
   if (rank == 0) {
      errno = 0;
      std::ifstream configFileStream{configFile};
      FatalIf(
            configFileStream.fail(),
            "ConfigFileArguments unable to open \"%s\" for reading: %s\n",
            configFile.c_str(),
            strerror(errno));
      configFileStream.seekg(0, std::ios_base::end);
      fileSize = (unsigned int)configFileStream.tellg();
      configContents.resize(fileSize);
      configFileStream.seekg(0, std::ios_base::beg);
      configFileStream.read(&configContents.at(0), fileSize);
      MPI_Bcast(&fileSize, 1, MPI_UNSIGNED, 0, communicator);
      MPI_Bcast(&configContents.at(0), (int)fileSize, MPI_CHAR, 0, communicator);
   }
   else {
      MPI_Bcast(&fileSize, 1, MPI_UNSIGNED, 0, communicator);
      configContents.resize(fileSize);
      MPI_Bcast(&configContents.at(0), (int)fileSize, MPI_CHAR, 0, communicator);
   }
   std::istringstream configStream{configContents};
   Arguments::resetState(configStream, allowUnrecognizedArguments);
}

} /* namespace PV */
