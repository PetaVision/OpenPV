/*
 * PVArguments.hpp
 *
 *  Created on: Sep 21, 2015
 *      Author: pschultz
 */

#ifndef ARGUMENTS_HPP_
#define ARGUMENTS_HPP_

#include "io/ConfigParser.hpp"
#include "io/Configuration.hpp"
#include <string>

namespace PV {

/**
 * A class for parsing a configuration and storing the results.
 * Internally, the Arguments object contains a require-return flag,
 * an output path string, a params file string, a log file string,
 * a gpu devices string, a working directory string, a checkpointRead directory
 * string, a restart flag, an unsigned integer indicating a random seed,
 * and integers indicating the number of threads, the number of MPI rows,
 * the number of MPI columns, and the batch width.
 * After initialization, individual settings can be modified with set-methods,
 * or reset to the original settings.
 *
 * It is an error to set both the restart flag and the checkpointRead directory
 * string.
 * Arguments does not check whether directory strings point at existing
 * directories, or do any other sanity checking of the arguments.
 * Typically under MPI, each mpi process will call PV_Init with the same
 * arguments, and each process's PV_Init object creates a Arguments object
 * that it, HyPerCol, and other objects can use to get the configuration
 * arguments.
 */
class Arguments {
  public:
   /**
    * The constructor for Arguments. The arguments are passed to initialize().
    */
   Arguments(std::istream &configStream, bool allowUnrecognizedArguments);

   /**
    * The destructor for Arguments.
    */
   virtual ~Arguments();

   /**
    * Returns the type for the given string:
    * unrecognized, boolean, integer, unsigned, string, or optional integer.
    */
   Configuration::ConfigurationType getType(std::string const &name) const {
      return mCurrentConfig.getType(name);
   }

   /**
    * Returns the value of the specified Boolean value in the current
    * configuration. Throws an invalid_argument exception if the input
    * argument does not refer to a Boolean.
    */
   bool const &getBooleanArgument(std::string const &name) const {
      return mCurrentConfig.getBooleanArgument(name);
   }

   /**
    * Returns the value of the specified integer in the current configuration.
    * Throws an invalid_argument exception if the input argument does not
    * refer to an integer.
    */
   int const &getIntegerArgument(std::string const &name) const {
      return mCurrentConfig.getIntegerArgument(name);
   }

   /**
    * Returns the value of the specified unsigned integer in the current
    * configuration. Throws an invalid_argument exception if the input
    * argument does not refer to an unsigned integer.
    */
   unsigned int const &getUnsignedIntArgument(std::string const &name) const {
      return mCurrentConfig.getUnsignedIntArgument(name);
   }

   /**
    * Returns the value of the specified string in the current configuration.
    * Throws an invalid_argument exception if the input argument does not
    * refer to a string in the configruation.
    */
   std::string const &getStringArgument(std::string const &name) const {
      return mCurrentConfig.getStringArgument(name);
   }

   /**
    * Returns the value of the specified optional integer in the current
    * configuration. Throws an invalid_argument exception if the input
    * argument does not refer to an optional integer.
    */
   Configuration::IntOptional const &getIntOptionalArgument(std::string const &name) const {
      return mCurrentConfig.getIntOptionalArgument(name);
   }

   /**
    * Sets the indicated Boolean value in the current configuration to the
    * indicated value. If the configuration does not have a Boolean
    * with the indicated name, the method does nothing.
    */
   bool setBooleanArgument(std::string const &name, bool const &value) {
      return mCurrentConfig.setBooleanArgument(name, value);
   }

   /**
    * Sets the indicated integer in the current configuration to the
    * indicated value. If the configuration does not have an integer
    * with the indicated name, the method does nothing.
    */
   bool setIntegerArgument(std::string const &name, int const &value) {
      return mCurrentConfig.setIntegerArgument(name, value);
   }

   /**
    * Sets the indicated unsigned integer in the current configuration to the
    * indicated value. If the configuration does not have an unsigned integer
    * with the indicated name, the method does nothing.
    */
   bool setUnsignedIntArgument(std::string const &name, unsigned int const &value) {
      return mCurrentConfig.setUnsignedIntArgument(name, value);
   }

   /**
    * Sets the indicated string in the current configuration to the
    * indicated value. If the configuration does not have a string
    * with the indicated name, the method does nothing.
    */
   bool setStringArgument(std::string const &name, std::string const &value) {
      return mCurrentConfig.setStringArgument(name, value);
   }

   /**
    * Sets the indicated optional integer in the current configuration to the
    * indicated value. If the configuration does not have an optional integer
    * with the indicated name, the method does nothing.
    */
   bool setIntOptionalArgument(std::string const &name, Configuration::IntOptional const &value) {
      return mCurrentConfig.setIntOptionalArgument(name, value);
   }

   /**
    * Sets the object's state based on the given input stream.
    * The allowUnrecognizedArguments flag has the same effect as in the
    * constructor.
    *
    * This method creates a ConfigParser object from its arguments,
    * and sets the Arguments data members according to the ConfigParser's
    * get-methods. See ConfigParser::initialize() for a description of
    * how the configuration stream is parsed.
    */
   void resetState(std::istream &configStream, bool allowUnrecognizedArguments);

   /**
    * Resets all member variables to their state at the time the object was
    * instantiated or reinitialized using resetState(istream, bool).
    * That is, the arguments are retrieved from the ConfigParser arguments,
    * and the effect of any set-method that had been called since then is
    * discarded.
    */
   void resetState();

   /**
    * Prints the effective command line based on the arguments used in
    * instantiation, and any set-methods used since then.
    */
   int printState() const;

  protected:
   /**
    * The constructor that should be called by classes derived from Arguments.
    * It calls initialize_base() but not initialize. The derived class's
    * own initialization should call Arguments::initialize().
    */
   Arguments() { initialize_base(); }

   /**
    * initialize() is called by the constructor, and calls
    * the resetState(istream, bool) method to set the data members.
    */
   int initialize(std::istream &configStream, bool allowUnrecognizedArguments);

  private:
   /**
    * initialize_base() is called by the constructor to initialize the internal
    * variables
    * to false for flags, zero for integers, and nullptr for strings.
    */
   int initialize_base();

   // Member variables
  private:
   ConfigParser *mConfigFromStream = nullptr;
   Configuration mCurrentConfig;
};

} /* namespace PV */

#endif /* ARGUMENTS_HPP_ */
