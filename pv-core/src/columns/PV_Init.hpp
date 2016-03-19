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
    */
   int initialize();

   /**
    * getParams() returns a pointer to the PVParams object created from the params file.
    */
   PVParams * getParams(){return params;}

   /**
    * setParams(paramsFile) creates the PVParams object for the given params file,
    * deleting the existing params object if it already exists.  It also sets the
    * params file returned by getArguments()->getParamsFile().
    */
   int setParams(char const * paramsFile);

   InterColComm * getComm(){return icComm;}
   int getWorldRank(){
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

private:
   int initSignalHandler();
   int initMaxThreads();
   int commInit(int* argc, char*** argv);
   int commFinalize();
   //int getNBatchValue(char* infile);
   PVParams * params;
   PV_Arguments * arguments;
   int initialized;
   int maxThreads;
   InterColComm * icComm;
};

}

#endif 
