/*
 * GenerativeConn.cpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#include "GenerativeConn.hpp"

namespace PV {

GenerativeConn::GenerativeConn() {
   initialize_base();
}  // end of GenerativeConn::GenerativeConn()

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *)

int GenerativeConn::initialize_base() {
   plasticityFlag = true; // Default value; override in params
   weightUpdatePeriod = 1;   // Default value; override in params

   nonnegConstraintFlag = false;
   return PV_SUCCESS;
   // Base class constructor calls base class initialize_base
   // so derived class initialize_base doesn't need to.
}

int GenerativeConn::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerConn::initialize(name, hc);
   return status;
}

int GenerativeConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_nonnegConstraintFlag(ioFlag);
   return status;
}

// TODO: make sure code works in non-shared weight case
void GenerativeConn::ioParam_sharedWeights(enum ParamsIOFlag ioFlag) {
   sharedWeights = true;
   if (ioFlag == PARAMS_IO_READ) {
      fileType = PVP_KERNEL_FILE_TYPE;
      parent->parameters()->handleUnnecessaryParameter(name, "sharedWeights", true/*correctValue*/);
   }
}

void GenerativeConn::ioParam_numAxonalArbors(enum ParamsIOFlag ioFlag) {
   HyPerConn::ioParam_numAxonalArbors(ioFlag);
   if (ioFlag == PARAMS_IO_READ && numAxonalArborLists!=1) {
      if (parent->columnId()==0) {
         fprintf(stderr, "GenerativeConn \"%s\" error: GenerativeConn has not been updated to support multiple arbors.\n", name);
      }
#ifdef PV_USE_MPI
      MPI_Barrier(parent->icCommunicator()->communicator());
#endif
      exit(EXIT_FAILURE);
   }
}

void GenerativeConn::ioParam_nonnegConstraintFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nonnegConstraintFlag", &nonnegConstraintFlag, false);
}

int GenerativeConn::allocateDataStructures() {
   int status = HyPerConn::allocateDataStructures();
   return status;
}

int GenerativeConn::update_dW(int axonID) {
   int status;
   status = defaultUpdate_dW(axonID);
   return status;
}  // end of GenerativeConn::update_dW(int);

int GenerativeConn::updateWeights(int axonID) {
   const int numPatches = getNumDataPatches();
   for( int k=0; k<numPatches; k++ ) {
      pvwdata_t * wdata = get_wDataHead(axonID, k);
      pvwdata_t * dwdata = get_dwDataHead(axonID, k);
      for( int y = 0; y < nyp; y++ ) {
         for( int x = 0; x < nxp; x++ ) {
            for( int f = 0; f < nfp; f++ ) {
               int idx = f*sfp + x*sxp + y*syp;
               wdata[idx] += dwdata[idx];
               if( nonnegConstraintFlag && wdata[idx] < 0) wdata[idx] = 0;
            }
         }
      }
   }
   return PV_SUCCESS;
}  // end of GenerativeConn::updateWeights(int)

}  // end of namespace PV block
