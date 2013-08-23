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

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, InitWeights *weightInit) {
   initialize_base();
   initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
//   HyPerLayer *, HyPerLayer *, int, const char *)

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

int GenerativeConn::initialize(const char * name, HyPerCol * hc,
      const char * pre_layer_name, const char * post_layer_name,
      const char * filename, InitWeights *weightInit) {
   KernelConn::initialize(name, hc, pre_layer_name, post_layer_name, filename, weightInit);

   return PV_SUCCESS;
}

int GenerativeConn::setParams(PVParams * params) {
   int status = KernelConn::setParams(params);
   readRelaxation(params);
   readNonnegConstraintFlag(params);
   readImprintingFlag(params);
   readWeightDecayFlag(params);
   readWeightDecayRate(params);
   readWeightNoiseLevel(params);
   return status;
}

void GenerativeConn::readNumAxonalArbors(PVParams * params) {
   KernelConn::readNumAxonalArbors(params);
   if (numAxonalArborLists!=1) {
      if (parent->columnId()==0) {
         fprintf(stderr, "GenerativeConn \"%s\" error: GenerativeConn has not been updated to support multiple arbors.\n", name);
      }
      MPI_Barrier(parent->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

void GenerativeConn::readRelaxation(PVParams * params) {
   relaxation = params->value(name, "relaxation", 1.0f);
}

void GenerativeConn::readNonnegConstraintFlag(PVParams * params) {
   nonnegConstraintFlag = params->value(name, "nonnegConstraintFlag", 0.0f) != 0.0; // default is not to constrain nonnegative.
}

void GenerativeConn::readImprintingFlag(PVParams * params) {
   imprintingFlag = params->value(name,"imprintingFlag", 0.f) != 0; // default is no imprinting
   if( imprintingFlag ) imprintCount = 0;
}

void GenerativeConn::readWeightDecayFlag(PVParams * params) {
   weightDecayFlag = params->value(name, "weightDecayFlag", 0.0f) != 0.0f;
}

void GenerativeConn::readWeightDecayRate(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "weightDecayFlag"));
   if( weightDecayFlag ) {
      weightDecayRate = params->value(name, "weightDecayRate", 0.0f);
   }
}

void GenerativeConn::readWeightNoiseLevel(PVParams * params) {
   assert(!params->presentAndNotBeenRead(name, "weightDecayFlag"));
   if( weightDecayFlag ) {
      weightNoiseLevel = params->value(name, "weightNoiseLevel", 0.0f);
   }
}

int GenerativeConn::allocateDataStructures() {
   int status = KernelConn::allocateDataStructures();
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
         const pvdata_t * patch_wData = get_wDataHead(axonID, p);
         pvdata_t * patch_dwData = get_dwDataHead(axonID, p);
         for(int k=0; k<nxp*nyp*nfp; k++) {
            pvdata_t decayterm = patch_wData[k];
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
         pvdata_t * dataPatch = get_wDataHead(0,p);
         dataPatch[imprintCount] = preSynapticLayer()->getLayerData(getDelays()[0])[p];
      }
      imprintCount++;
      return PV_SUCCESS;
   }
   for( int k=0; k<numPatches; k++ ) {
      pvdata_t * wdata = get_wDataHead(axonID, k);
      pvdata_t * dwdata = get_dwDataHead(axonID, k);
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

#ifdef OBSOLETE // Marked obsolete April 16, 2013.  Implementing the new NormalizeBase class hierarchy
int GenerativeConn::initNormalize() {
   PVParams * params = parent->parameters();
   normalize_flag = params->value(name, "normalize", normalize_flag) != 0;
   if( !normalize_flag ) return PV_SUCCESS;
   normalizeMethod = (int) params->value(name, "normalizeMethod", 0.f); // default is no constraint
   switch( normalizeMethod ) {
   case 0:
      break;
   case 1:
      KernelConn::initNormalize();
      break;
   case 2: // fallthrough is intentional
   case 3:
   case 4:
      normalizeConstant = params->value(name, "normalizeConstant", 1.0f);
      break;
   default:
      fprintf(stderr,"Connection \"%s\": Unrecognized normalizeMethod %d.  Using normalizeMethod = 1 (KernelConn's normalizeWeights).\n", this->getName(), normalizeMethod);
      KernelConn::initNormalize();
      break;
   }
   return PV_SUCCESS;
}

int GenerativeConn::normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId) {
   int status = PV_SUCCESS;
   int neuronsperpatch;
   switch( normalizeMethod ) {
   case 0:
      break;
   case 1:
      status = KernelConn::normalizeWeights(patches, dataStart, numPatches, arborId);
      break;
   case 2:
      neuronsperpatch = nxp*nyp*nfp;
      for( int n=0; n<neuronsperpatch; n++ ) {
         pvdata_t s = 0;
         for( int k=0; k<numPatches; k++ ) {
            pvdata_t d = dataStart[arborId][k*nxp*nyp*nfp+n];
            s += d*d;
         }
         s = sqrt(s);
         for( int k=0; k<numPatches; k++ ) {
            dataStart[arborId][k*nxp*nyp*nfp+n] *= normalizeConstant/s;
         }
      }
      break;
   case 3:
      neuronsperpatch = nxp*nyp*nfp;
      for( int k=0; k<numPatches; k++ ) {
         pvdata_t s = 0;
         for( int n=0; n<neuronsperpatch; n++ ) {
            pvdata_t d = dataStart[arborId][k*nxp*nyp*nfp+n];
            s += d*d;
         }
         s = sqrt(s);
         for( int n=0; n<neuronsperpatch; n++ ) {
            dataStart[arborId][k*nxp*nyp*nfp+n] *= normalizeConstant/s;
         }
      }
      break;
   case 4:
      neuronsperpatch = nxp*nyp*nfp;
      for( int k=0; k<numPatches; k++ ) {
         pvdata_t * dataHead = get_wDataHead(0, k);
         for( int n=0; n<neuronsperpatch; n++ ) {
            if( dataHead[n] > normalizeConstant ) dataHead[n] = normalizeConstant;
         }
      }
      break;
   default:
      assert(false); // This possibility was eliminated in initNormalize().
      break;
   }
   return status;
}  // end of GenerativeConn::normalizeWeights
#endif // OBSOLETE

}  // end of namespace PV block
