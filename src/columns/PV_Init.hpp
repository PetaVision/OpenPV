/*
 * PV_Init.hpp
 *
 *  Created on: Jul 31, 2015
 *      Author: slundquist
 */

#ifndef PV_INIT_HPP_
#define PV_INIT_HPP_

#include <arch/mpi/mpi.h>
#include <columns/Arguments.hpp>
#include <columns/Factory.hpp>
#include <columns/KeywordHandler.hpp>
#include <io/PVParams.hpp>
#include <iostream>

namespace PV {

/**
 * PV_Init is an object that initializes MPI and parameters to pass to the
 * HyPerCol
 */
class PV_Init {
  public:
   /**
    * The constructor creates an Arguments object from the input arguments
    * and if MPI has not already been initialized, calls MPI_Init.
    * Note that it does not call initialize, so the PVParams and Communicator
    * objects are not initialized on instantiation.
    * On instantiation, the create() method will recognize all core PetaVision
    * groups (ANNLayer, HyPerConn, etc.).  To add additional known groups,
    * see the registerKeyword method.
    */
   PV_Init(int *argc, char **argv[], bool allowUnrecognizedArguments);
   /**
    * Destructor calls MPI_Finalize
    */
   virtual ~PV_Init();

   /**
    * initialize(void) creates the PVParams and Communicator objects from
    * the existing arguments.  If the paramsFile (-p) argument is not set,
    * params is kept at null, and the params file can be set later using the
    * setParams() method.
    *
    * initialize() is called by the PV_Init constructor, but it is
    * also permissible to call initialize again, in which case the
    * PVParams and Communicator objects are deleted and recreated,
    * based on the current state of the arguments.
    */
   int initialize();

   // Below are get-methods for retrieving command line arguments from the
   // Arguments object.  Note that there are both getParams and
   // getParamsFile methods.

   /**
    * Returns a copy of the args array.  It uses malloc and strdup, so the caller
    * is responsible for freeing getArgs()[k] for each k and for freeing
    * getArgs()
    * (the simplest way to free all the memory at once is to call the
    * static method Argument::freeArgs)
    * The length of the returned array is argc+1, and getArgs()[argc] is NULL.
    */
   char **getArgsCopy() const;

   /**
    * Deallocates an array, assuming it was created by a call to getArgsCopy().
    * It frees argv[0], argv[1], ..., argv[argc-1], and then frees argv.
    */
   static void freeArgs(int argc, char **argv);

   /**
    * Returns the argc argument passed to the constructor.
    */
   int getNumArgs() const { return mArgC; }

   /**
    * Returns true if the require-return flag was set.
    */
   char const *getProgramName() const { return mArgV[0]; }

   /**
    * Returns a pointer to the Arguments. Declared const, so the arguments
    * cannot be changed using the result of this function.
    */
   Arguments const *getArguments() const { return arguments; }

   bool const &getBooleanArgument(std::string const &name) const {
      return arguments->getBooleanArgument(name);
   }

   int const &getIntegerArgument(std::string const &name) const {
      return arguments->getIntegerArgument(name);
   }

   unsigned int const &getUnsignedIntArgument(std::string const &name) const {
      return arguments->getUnsignedIntArgument(name);
   }

   std::string const &getStringArgument(std::string const &name) const {
      return arguments->getStringArgument(name);
   }

   Configuration::IntOptional const &getIntOptionalArgument(std::string const &name) const {
      return arguments->getIntOptionalArgument(name);
   }

   /**
    * getParams() returns a pointer to the PVParams object created from the
    * params file.
    */
   PVParams *getParams() { return params; }

   /**
    * Prints the effective command line based on the argc/argv arguments used in
    * instantiation,
    * and any set-methods used since then.
    */
   void printState() const { arguments->printState(); }

   // Below are set-methods for changing changing command line arguments
   // stored in the Arguments object, and doing any necessary
   // operations required by the change.

   bool setBooleanArgument(std::string const &name, bool const &value) {
      return arguments->setBooleanArgument(name, value);
   }

   bool setIntegerArgument(std::string const &name, int const &value) {
      return arguments->setIntegerArgument(name, value);
   }

   bool setUnsignedIntArgument(std::string const &name, unsigned int const &value) {
      return arguments->setUnsignedIntArgument(name, value);
   }

   bool setStringArgument(std::string const &name, std::string const &value) {
      return arguments->setStringArgument(name, value);
   }

   bool setIntOptionalArgument(std::string const &name, Configuration::IntOptional const &value) {
      return arguments->setIntOptionalArgument(name, value);
   }

   /**
    * setParams(paramsFile) updates the params file stored in the arguments,
    * and calls PV_Init::initialize, which deletes the previous params object
    * if it exists, and creates the new one.
    * Return value is PV_SUCCESS or PV_FAILURE.
    * If the routine fails, the params are unchanged.
    */
   int setParams(char const *paramsFile);

   /**
    * Sets the log file.  If the string argument is null, logging returns to the
    * default streams (probably cout and cerr).  The previous log file,
    * if any, is closed; and the new file is opened in write mode.
    * Return value is PV_SUCCESS or PV_FAILURE.
    * If the routine fails, the logging streams remain unchanged.
    */
   int setLogFile(char const *val, bool appendFlag = false);

   /**
    * Sets the number of rows, columns, and batch elements.
    * If any of these values are zero, they will be inferred
    * from the other values (as if the relevant command line option was absent.)
    * If any of the arguments are negative, the corresponding values are
    * left unchanged.
    * initialize() is called, which deletes the existing Communicator
    * and creates a new one.
    * Returns PV_SUCCESS if successful; exits on an error if it fails.
    */
   int setMPIConfiguration(int rows, int columns, int batchwidth);

   /**
    * Resets all member variables to their state at the time the object was
    * instantiated. That is, the arguments in the original argv are parsed
    * again, and the effect of any set-method that had been called is
    * discarded.  Any previous pointers returned by get-methods, except
    * for getArgs or getArgsCopy are no longer valid.
    * Always returns PV_SUCCESS.  If the routine fails, it exits with an error.
    */
   int resetState();

   Communicator *getCommunicator() { return mCommunicator; }

   int getWorldRank() const {
      if (mCommunicator) {
         return mCommunicator->globalCommRank();
      }
      else {
         int rank = 0;
         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
         return rank;
      }
   }

   int getWorldSize() {
      if (mCommunicator) {
         return mCommunicator->globalCommSize();
      }
      else {
         int size = 0;
         MPI_Comm_size(MPI_COMM_WORLD, &size);
         return size;
      }
   }

   bool isExtraProc() { return mCommunicator->isExtraProc(); }

   /**
    * If using PV_USE_OPENMP_THREADS, returns the value returned by
    * omp_get_max_threads() when the
    * PV_Init object was instantiated.
    * Note that this value is NOT divided by the number of MPI processes.
    * If not using PV_USE_OPENMP_THREADS, returns 1.
    */
   int getMaxThreads() const { return maxThreads; }

   /**
    * The method to add a new object type to the PV_Init object's class factory.
    * keyword is the string that labels the object type, matching the keyword
    * used in params files.
    * creator is a pointer to a function that takes a name and a HyPerCol
    * pointer, and
    * creates an object of the corresponding keyword, with the given name and
    * parent HyPerCol.
    * The function should return a pointer of type BaseObject, created with the
    * new operator.
    */
   int registerKeyword(char const *keyword, ObjectCreateFn creator);

  private:
   int initSignalHandler();
   int initMaxThreads();
   int commInit(int *argc, char ***argv);

   /**
    * Makes sure that the Factory singleton is initialized, and registers the core keywords to
    * the factory, by calling registerCoreKeywords(), defined in the CoreKeywords.hpp file.
    */
   void initFactory();

   /**
    * A method used internally by initialize() to set the streams that will
    * be used by InfoLog(), WarnLog(), etc.
    * If the logFile is a path, the root process writes to that path and the
    * nonroot process will write to a path modified by inserting _<rank>
    * before the extension, or at the end if the path has no extension.
    * If the logFile is null, all processes write to the standard output and
    * error streams.
    *
    * After setting the log file streams, initLogFile() writes the time stamp
    * to InfoLog() and calls Arguments::printState(), which writes the
    * effective
    * command line to InfoLog().
    */
   void initLogFile(bool appendFlag);

   /**
    * A method used internally by initialize() and setParams() to create the
    * PVParams object
    * from the params file set in the arguments.
    * If the arguments has the params file set, it creates the PVParams object
    * and returns success;
    * otherwise it returns failure and leaves the value of the params data member
    * unchanged.
    */
   int createParams();

   /**
    * Sends a timestamp and the effective command line to the InfoLog stream.
    * The effective command line is based on the current state of the arguments
    * data member.
    */
   void printInitMessage();

   int commFinalize();

   int mArgC = 0;
   std::vector<char const *> mArgV;
   PVParams *params;
   Arguments *arguments;
   int maxThreads;
   Communicator *mCommunicator;
};
}

#endif
