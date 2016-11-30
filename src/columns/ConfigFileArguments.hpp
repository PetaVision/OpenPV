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
 * The arguments are passed to initialize(), which calls resetState().
 */
class ConfigFileArguments : public Arguments {
  public:
   /**
    * The constructor for ConfigFileArguments.
    * The given config file is opened as an ifstream and passed
    * to Arguments::initialize().
    */
   ConfigFileArguments(std::string const &configFile, MPI_Comm communicator, bool allowUnrecognizedArguments);

   /**
    * The destructor for ConfigFileArguments.
    */
   virtual ~ConfigFileArguments() {}

   /**
    * resetState is called during instantiation, and can also be called as a
    * public method to discard the results of any set-methods and to use a new
    * file for the base configuration.
    *
    * The root process of the communicator reads the file specified by
    * configFile into a string, and broadcasts it to the other processes of
    * the communicator.  That string is then converted to an input stringstream
    * which is passed to Arguments::resetState.
    */
   void resetState(std::string const &configFile, MPI_Comm communicator, bool allowUnrecognizedArguments);

  protected:
   /**
    * Called by the ConfigFileArguments constructor. It calls resetState with its arguments.
    */
   int initialize(std::string const &configFile, MPI_Comm communicator, bool allowUnrecognizedArguments);

  private:
   /**
    * initialize_base() is called by the constructor to initialize the internal
    * variables to false for flags, zero for integers, and nullptr for strings.
    */
   int initialize_base();
};

} /* namespace PV */

#endif /* CONFIGFILEARGUMENTS_HPP_ */
