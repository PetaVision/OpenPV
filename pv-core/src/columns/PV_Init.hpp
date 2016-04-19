/*
 * PV_Init.hpp
 *
 *  Created on: Jul 31, 2015
 *      Author: slundquist
 */

#ifndef PV_INIT_HPP_
#define PV_INIT_HPP_

#include <iostream>
#include <arch/mpi/mpi.h>
#include <io/PVParams.hpp>
#include <io/io.h>
#include <columns/PV_Arguments.hpp>
#include <columns/Factory.hpp>

namespace PV {

/**
 * PV_Init is an object that initializes MPI and parameters to pass to the HyPerCol
 */
class PV_Init {
public:
   /**
    * The constructor creates a PV_Arguments object from the input arguments
    * and if MPI has not already been initialized, calls MPI_Init.
    * Note that it does not call initialize, so the PVParams and InterColComm
    * objects are not initialized on instantiation.
    * On instantiation, the create() method will recognize all core PetaVision
    * groups (ANNLayer, HyPerConn, etc.).  To add additional known groups,
    * see the registerKeyword method.
    */
   PV_Init(int* argc, char ** argv[], bool allowUnrecognizedArguments);
   /**
    * Destructor calls MPI_Finalize
    */
   virtual ~PV_Init();

   /**
    * initialize(void) creates the PVParams and InterColComm objects from
    * the existing arguments.  If the paramsFile (-p) argument is not set,
    * params is kept at null, and the params file can be set later using the
    * setParams() method.
    *
    * initialize() is called by the PV_Init constructor, but it is
    * also permissible to call initialize again, in which case the
    * PVParams and InterColComm objects are deleted and recreated,
    * based on the current state of the arguments.
    */
   int initialize();

   /**
    * getParams() returns a pointer to the PVParams object created from the params file.
    */
   PVParams * getParams(){return params;}

   /**
    * setParams(paramsFile) creates the PVParams object for the given
    * params file, deleting the existing params object if it already exists.
    * It also calls PV_Init::initialize if it hasn't been called, and updates
    * the params updates the params file returned by
    * getArguments()->getParamsFile().
    */
   int setParams(char const * paramsFile);

   InterColComm * getComm(){return icComm;}
   int getWorldRank() const {
      if(icComm){
         return icComm->globalCommRank();
      }
      else{
         int rank = 0;
         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
         return rank;
      }
   }
   int getWorldSize(){
      if(icComm){
         return icComm->globalCommSize();
      }
      else{
         int size = 0;
         MPI_Comm_size(MPI_COMM_WORLD, &size);
         return size;
      }
   }
   int isExtraProc(){return icComm->isExtraProc();}
   int getInit(){return initialized;}

   /**
    * If using PV_USE_OPENMP_THREADS, returns the value returned by omp_get_max_threads() when the PV_Init object was instantiated.
    * Note that this value is NOT divided by the number of MPI processes.
    * If not using PV_USE_OPENMP_THREADS, returns 1.
    */
   int getMaxThreads() const {return maxThreads; }

   /**
    * Returns the PV_Arguments object holding the parsed values of the command line arguments.
    */
   PV_Arguments * getArguments() { return arguments; }

   /**
    * The method to add a new object type to the PV_Init object's class factory.
    * keyword is the string that labels the object type, matching the keyword used in params files.
    * creator is a pointer to a function that takes a name and a HyPerCol pointer, and
    * creates an object of the corresponding keyword, with the given name and parent HyPerCol.
    * The function should return a pointer of type BaseObject, created with the new operator.
    */
   int registerKeyword(char const * keyword, ObjectCreateFn creator);

   /**
    * The method to create an object (layer, connection, weight initializer, weight normalizer,
    * or probe) based on keyword and name, and add it to the given HyPerLayer.
    */
   BaseObject * create(char const * keyword, char const * name, HyPerCol * hc) const;

   /**
    * Forms a HyPerCol, and adds layers, probes, and connections to it based on the PV_Init
    * object's current params.  If any of the groups in params fails to build, the HyPerCol
    * is deleted and build() returns NULL.  An error message indicates which params group failed.
    */
   HyPerCol * build();

   /**
    * This function turns the buildandrunDeprecationWarning flag on.
    * It is used by the deprecated functions in buildandrun.cpp to
    * manage printing the deprecation warning once but not more than
    * once if a deprecated function calls another deprecated function.
    * When the deprecated buildandrun functions (which were deprecated Mar 24, 2016)
    * become obsolete, this and related PV_Init functions can be removed.
    */
   void setBuildAndRunDeprecationWarning() { buildandrunDeprecationWarning = true; }

   /**
    * This function turns the buildandrunDeprecationWarning flag off.
    * It is used by the deprecated functions in buildandrun.cpp to
    * manage printing the deprecation warning once but not more than
    * once if a deprecated function calls another deprecated function.
    * When the deprecated buildandrun functions (which were deprecated Mar 24, 2016)
    * become obsolete, this and related PV_Init functions can be removed.
    */
   void clearBuildAndRunDeprecationWarning() { buildandrunDeprecationWarning = false; }

   /**
    * This function prints a warning if the buildandrunDeprecationWarning is on.
    * It is used by the deprecated functions in buildandrun.cpp to
    * manage printing the deprecation warning once but not more than
    * once if a deprecated function calls another deprecated function.
    * When the deprecated buildandrun functions (which were deprecated Mar 24, 2016)
    * become obsolete, this and related PV_Init functions can be removed.
    */
   void printBuildAndRunDeprecationWarning(char const * functionName, char const * functionSignature);

private:
   int initSignalHandler();
   int initMaxThreads();
   int commInit(int* argc, char*** argv);

   /**
    * A method used internally by initialize() and setParams() to create the PVParams object
    * from the params file set in the arguments.
    * If the arguments has the params file set, it creates the PVParams object and returns success;
    * otherwise it returns failure and leaves the value of the params data member unchanged.
    */
   int createParams();

   int commFinalize();
   PVParams * params;
   PV_Arguments * arguments;
   int initialized;
   int maxThreads;
   InterColComm * icComm;
   Factory * factory;
   bool buildandrunDeprecationWarning;
};

}

#endif 
