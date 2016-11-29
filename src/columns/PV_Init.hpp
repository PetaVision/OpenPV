/*
 * PV_Init.hpp
 *
 *  Created on: Jul 31, 2015
 *      Author: slundquist
 */

#ifndef PV_INIT_HPP_
#define PV_INIT_HPP_

#include <arch/mpi/mpi.h>
#include <columns/Factory.hpp>
#include <columns/Arguments.hpp>
#include <io/PVParams.hpp>
#include <io/io.hpp>
#include <iostream>

namespace PV {

class HyPerCol; // Included only to allow obsolete (as of Jul 19, 2016)
// HyPerCol* PV_Init::build()
// method to print an error message.

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
    * static method PV_Argument::freeArgs)
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
    * Returns the output path string.
    */
   char const *getOutputPath() const { return arguments->getOutputPath().c_str(); }

   /**
    * Returns the params file string.
    */
   char const *getParamsFile() const { return arguments->getParamsFile().c_str(); }

   /**
    * getParams() returns a pointer to the PVParams object created from the
    * params file.
    */
   PVParams *getParams() { return params; }

   /**
    * Returns the log file string.
    */
   char const *getLogFile() const { return arguments->getLogFile().c_str(); }

   /**
    * Returns the gpu devices string.
    */
   char const *getGPUDevices() const { return arguments->getGPUDevices().c_str(); }

   /**
    * Returns the random seed.
    */
   unsigned int getRandomSeed() const { return arguments->getRandomSeed(); }

   /**
    * Returns the working directory string.
    */
   char const *getWorkingDir() const { return arguments->getWorkingDir().c_str(); }

   /**
    * Returns true if the restart flag was set.
    */
   bool getRestartFlag() const { return arguments->getRestartFlag(); }

   /**
    * Returns the checkpointRead directory string.
    */
   char const *getCheckpointReadDir() const { return arguments->getCheckpointReadDir().c_str(); }

   /**
    * Returns the useDefaultNumThreads flag.
    */
   bool getUseDefaultNumThreads() const { return arguments->getUseDefaultNumThreads(); }

   /**
    * Returns the number of threads.
    */
   int getNumThreads() const { return arguments->getNumThreads(); }

   /**
    * Returns the number of rows.
    */
   int getNumRows() const { return arguments->getNumRows(); }

   /**
    * Returns the number of columns.
    */
   int getNumColumns() const { return arguments->getNumColumns(); }

   /**
    * Returns the batch width.
    */
   int getBatchWidth() const { return arguments->getBatchWidth(); }

   /**
    * Returns true if the dry-run flag was set.
    */
   bool getDryRunFlag() const { return arguments->getDryRunFlag(); }

   /**
    * Prints the effective command line based on the argc/argv arguments used in
    * instantiation,
    * and any set-methods used since then.
    */
   void printState() const { arguments->printState(); }

   // Below are set-methods for changing changing command line arguments
   // stored in the Arguments object, and doing any necessary
   // operations required by the change.

   /**
    * Sets the value of the require-return flag.  Always returns PV_SUCCESS.
    */
   int setRequireReturnFlag(bool val) {
      arguments->setRequireReturnFlag(val);
      return PV_SUCCESS;
   }

   /**
    * Sets the value of the output path string.
    */
   int setOutputPath(char const *val) {
      arguments->setOutputPath(val);
      return PV_SUCCESS;
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
    * Sets the value of the gpu devices string to a copy of the input argument.
    * Note that this only changes the string; it doesn't touch the GPUs.
    */
   int setGPUDevices(char const *val) {
      arguments->setGPUDevices(val);
      return PV_SUCCESS;
   }

   /**
    * Sets the value of the random seed to the input argument.
    * The return value is always PV_SUCCESS.
    */
   unsigned int setRandomSeed(unsigned int val) {
      arguments->setRandomSeed(val);
      return PV_SUCCESS;
   }

   /**
    * Sets the value of the working directory string to a copy of the input
    * argument.
    * Return value is PV_SUCCESS or PV_FAILURE.
    * If the routine fails, the working directory string remains unchanged.
    * TODO: PV_Init should handle the working directory, not HyPerCol.
    */
   int setWorkingDir(char const *val) {
      arguments->setWorkingDir(val);
      return PV_SUCCESS;
   }

   /**
    * Sets the value of the restart flag.
    * The return value is always PV_SUCCESS
    */
   int setRestartFlag(bool val) {
      arguments->setRestartFlag(val);
      return PV_SUCCESS;
   }

   /**
    * Sets the value of the checkpointRead directory string.
    * Return value is PV_SUCCESS or PV_FAILURE.
    * If the routine fails, the checkpointRead directory string is unchanged.
    * Note that this only changes the string; it doesn't examine the directory.
    */
   int setCheckpointReadDir(char const *val) {
      arguments->setCheckpointReadDir(val);
      return PV_SUCCESS;
   }

   /**
     * Turns on the useDefaultNumThreads flag, and sets numThreads to zero.
    * (Should be
    * maxthreads/numProcesses)
     * The return value is always PV_SUCCESS
     */
   int setUseDefaultNumThreads() {
      arguments->setUseDefaultNumThreads(true);
      return PV_SUCCESS;
   }

   /**
    * Sets the number of threads to the input argument.
    * Additionally, turns off the useDefaultNumThreads.
    * The return value is always PV_SUCCESS
    */
   int setNumThreads(int val) {
      arguments->setNumThreads(val);
      return PV_SUCCESS;
   }

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
    * Sets the dry-run flag to the new argument.
    * The return value is always PV_SUCCESS
    */
   int setDryRunFlag(bool val) {
      arguments->setDryRunFlag(val);
      return PV_SUCCESS;
   }

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

   int isExtraProc() { return mCommunicator->isExtraProc(); }

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

   /**
    * Obsolete.  Use createHyPerCol defined in HyPerCol.cpp instead.
    */
   HyPerCol *build() {
      Fatal() << "PV_Init::build is obsolete.  " // marked obsolete July 19, 2016.
              << "Use hc=createHyPerCol(pv_init_ptr) instead of "
                 "hc=pv_init_ptr->build()\n";
      return nullptr;
   }

  private:
   int initSignalHandler();
   int initMaxThreads();
   int commInit(int *argc, char ***argv);

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
   char **mArgV = nullptr;
   PVParams *params;
   Arguments *arguments;
   int maxThreads;
   Communicator *mCommunicator;
};
}

#endif
