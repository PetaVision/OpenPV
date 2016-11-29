/*
 * ConfigFileArguments.hpp
 *
 *  Created on: Nov 28, 2016
 *      Author: pschultz
 */

#ifndef CONFIGFILEARGUMENTS_HPP_
#define CONFIGFILEARGUMENTS_HPP_

#include "Arguments.hpp"
#include "arch/mpi/mpi.h"

namespace PV {

/**
 * A subclass of Arguments whose constructor takes a filename as an argument.
 * An input filestream is created from the filename and passed to
 * Arguments::initialize().
 */
class ConfigFileArguments : public Arguments {
  public:
   /**
    * The constructor for ConfigFileArguments.
    * The given config file is opened as an ifstream and passed
    * to Arguments::initialize().
    */
   ConfigFileArguments(std::string const &configFile, MPI_Comm communicator, bool allowUnrecognizedArguments);

   /*
    * The destructor for ConfigFileArguments.
    */
   virtual ~ConfigFileArguments() {}

   void resetState(std::string const &configFile, MPI_Comm communicator, bool allowUnrecognizedArguments);

  protected:
   int initialize(std::string const &configFile, MPI_Comm communicator, bool allowUnrecognizedArguments);

  private:
   /**
    * initialize_base() is called by the constructor to initialize the internal
    * variables
    * to false for flags, zero for integers, and nullptr for strings.
    */
   int initialize_base();

  private:
   std::string mConfigContents;
};

} /* namespace PV */

#endif /* CONFIGFILEARGUMENTS_HPP_ */
