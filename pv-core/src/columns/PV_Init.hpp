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

namespace PV {


/**
 * PV_Init is an object that initializes MPI and parameters to pass to the HyPerCol
 */
class PV_Init {
public:
   /**
    * The constructor does not call initialize. This does call MPI_Init
    */
   PV_Init(int* argc, char ** argv[]);
   /**
    * Destructor calls MPI_Finalize
    */
   virtual ~PV_Init();
   /**
    * Initialize allocates the PVParams and InterColComm objects
    */
   int initialize(int argc, char* argv[]);
   /**
    * Sets the previously allocated PVParams and InterColComm objects
    */
   int initialize(PVParams* inparams, InterColComm* incomm);

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


private:
   int commInit(int* argc, char*** argv);
   int commFinalize();
   //int getNBatchValue(char* infile);
   PVParams * params;
   InterColComm * icComm;
   int initialized;
};

}

#endif 
