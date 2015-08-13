/*
 * PV_Init.hpp
 *
 *  Created on: Jul 31, 2015
 *      Author: slundquist
 */

#ifndef PV_INIT_HPP_
#define PV_INIT_HPP_

#include "../io/PVParams.hpp"
#include "../io/io.h"
//#include "InterColComm.hpp"

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
   int initialize(PVParams* inparams);

   PVParams * getParams(){return params;}
   //InterColComm * getComm(){return icComm;}
   int getWorldRank(){return worldRank;}
   int getWorldSize(){return worldSize;}
private:
   int commInit(int* argc, char*** argv);
   int commFinalize();
   PVParams * params;
   //InterColComm * icComm;
   int initialized;
   int worldRank;
   int worldSize;
};

}

#endif 
