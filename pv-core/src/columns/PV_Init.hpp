/*
 * PV_Init.hpp
 *
 *  Created on: Jul 31, 2015
 *      Author: slundquist
 */

#ifndef PV_INIT_HPP_
#define PV_INIT_HPP_

#include <iostream>
#include "../arch/mpi/mpi.h"
#include "../io/PVParams.hpp"
#include "../io/io.h"
#include "PV_Arguments.hpp"

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
    * the existing arguments.
    */
   int initialize();

   PVParams * getParams(){return params;}
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
    * Returns the PV_Arguments object holding the parsed values of the command line arguments.
    */
   PV_Arguments * getArguments() { return arguments; }

private:
   int initSignalHandler();
   int commInit(int* argc, char*** argv);
   int commFinalize();
   //int getNBatchValue(char* infile);
   PVParams * params;
   InterColComm * icComm;
   PV_Arguments * arguments;
   int initialized;
};

}

#endif 
