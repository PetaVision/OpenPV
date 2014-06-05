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

   relaxation = 1.0;
   nonnegConstraintFlag = false;
   normalizeMethod = 0;
   imprintingFlag = false;
   imprintCount = 0;
   weightDecayFlag = false;
   weightDecayRate = 0.0;
   weightNoiseLevel = 0.0;
   noise = NULL;
   return PV_SUCCESS;
   // Base class constructor calls base class initialize_base
   // so derived class initialize_base doesn't need to.
}

int GenerativeConn::initialize(const char * name, HyPerCol * hc) {
   int status = HyPerConn::initialize(name, hc);
   assert(!parent->parameters()->presentAndNotBeenRead(name, "imprintingFlag"));
   if( imprintingFlag ) imprintCount = 0;
   return status;
}

int GenerativeConn::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = HyPerConn::ioParamsFillGroup(ioFlag);
   ioParam_relaxation(ioFlag);
   ioParam_nonnegConstraintFlag(ioFlag);
   ioParam_imprintingFlag(ioFlag);
   ioParam_weightDecayFlag(ioFlag);
   ioParam_weightDecayRate(ioFlag);
   ioParam_weightNoiseLevel(ioFlag);
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

void GenerativeConn::ioParam_relaxation(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "relaxation", &relaxation, 1.0f);
}

void GenerativeConn::ioParam_nonnegConstraintFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "nonnegConstraintFlag", &nonnegConstraintFlag, false);
}

void GenerativeConn::ioParam_imprintingFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "imprintingFlag", &imprintingFlag, false);
}

void GenerativeConn::ioParam_weightDecayFlag(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "weightDecayFlag", &weightDecayFlag, false);
}

void GenerativeConn::ioParam_weightDecayRate(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "weightDecayFlag"));
   if (weightDecayFlag) {
      parent->ioParamValue(ioFlag, name, "weightDecayRate", &weightDecayRate, 0.0f);
   }
}

void GenerativeConn::ioParam_weightNoiseLevel(enum ParamsIOFlag ioFlag) {
   assert(!parent->parameters()->presentAndNotBeenRead(name, "weightDecayFlag"));
   if (weightDecayFlag) {
      parent->ioParamValue(ioFlag, name, "weightNoiseLevel", &weightNoiseLevel, 0.0f);
   }
}

int GenerativeConn::allocateDataStructures() {
   int status = HyPerConn::allocateDataStructures();
   if (weightDecayFlag) {
      // All processes should have the same seed.
      // We create a Random object with one RNG, seeded the same way.
      // Another approach would be to have a separate RNG for each data patch:
      // noise = new Random(parent, getNumDataPatches());
      // or even a separate RNG for each weight value:
      // noise = new Random(parent, getNumDataPatches()*nxp*nyp*nfp);
      // These would be helpful if parallelizing, but could require
      // the resulting rngArray to be large.
      noise = new Random(parent, 1);
   }
   return status;
}

int GenerativeConn::update_dW(int axonID) {
   int status;
   status = defaultUpdate_dW(axonID);
   if(weightDecayFlag) {
      for(int p=0; p<getNumDataPatches(); p++) {
         const pvwdata_t * patch_wData = get_wDataHead(axonID, p);
         pvwdata_t * patch_dwData = get_dwDataHead(axonID, p);
         for(int k=0; k<nxp*nyp*nfp; k++) {
            pvwdata_t decayterm = patch_wData[k];
            patch_dwData[k] += -weightDecayRate * decayterm;
            if (weightDecayFlag) patch_dwData[k] += weightNoiseLevel * noise->uniformRandom(0, -1.0f, -1.0f);
         }
      }
   }
   return status;
}  // end of GenerativeConn::update_dW(int);

int GenerativeConn::updateWeights(int axonID) {
   const int numPatches = getNumDataPatches();
   if( imprintingFlag && imprintCount < nfp ) {
      assert(nxp==1 && nyp==1 && numberOfAxonalArborLists()==1);
      for( int p=0; p<numPatches; p++ ) {
         pvwdata_t * dataPatch = get_wDataHead(0,p);
         dataPatch[imprintCount] = preSynapticLayer()->getLayerData(getDelays()[0])[p];
      }
      imprintCount++;
      return PV_SUCCESS;
   }
   for( int k=0; k<numPatches; k++ ) {
      pvwdata_t * wdata = get_wDataHead(axonID, k);
      pvwdata_t * dwdata = get_dwDataHead(axonID, k);
      for( int y = 0; y < nyp; y++ ) {
         for( int x = 0; x < nxp; x++ ) {
            for( int f = 0; f < nfp; f++ ) {
               int idx = f*sfp + x*sxp + y*syp;
               wdata[idx] += relaxation*dwdata[idx];
               if( nonnegConstraintFlag && wdata[idx] < 0) wdata[idx] = 0;
            }
         }
      }
   }
   return PV_SUCCESS;
}  // end of GenerativeConn::updateWeights(int)

}  // end of namespace PV block
