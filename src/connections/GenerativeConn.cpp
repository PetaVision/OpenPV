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
      HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights *weightInit) {
   initialize_base();
   initialize(name, hc, pre, post, filename, weightInit);
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
   return PV_SUCCESS;
   // Base class constructor calls base class initialize_base
   // so derived class initialize_base doesn't need to.
}

#ifdef OBSOLETE
int GenerativeConn::initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel) {
   return initialize(name, hc, pre, post, channel, NULL, NULL);
}
#endif // OBSOLETE

int GenerativeConn::initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights *weightInit) {
   PVParams * params = hc->parameters();
   relaxation = params->value(name, "relaxation", 1.0f);
   nonnegConstraintFlag = params->value(name, "nonnegConstraintFlag", 0.0f) != 0.0; // default is not to constrain nonnegative.
   imprintingFlag = params->value(name,"imprintingFlag", 0.f) != 0; // default is no imprinting
   if( imprintingFlag ) imprintCount = 0;
   weightDecayFlag = params->value(name, "weightDecayFlag", 0.0f) != 0.0f;
   if( weightDecayFlag ) {
      weightDecayRate = params->value(name, "weightDecayRate", 0.0f);
      weightNoiseLevel = params->value(name, "weightNoiseLevel", 0.0f);

   }
   KernelConn::initialize(name, hc, pre, post, filename, weightInit);

   //GenerativeConn has not been updated to support multiple arbors!
   assert(numberOfAxonalArborLists()==1);

   return PV_SUCCESS;
}

int GenerativeConn::update_dW(int axonID) {
   int status;
   status = defaultUpdate_dW(axonID);
   if(weightDecayFlag) {
      for(int p=0; p<getNumDataPatches(); p++) {
         const pvdata_t * patch_wData = get_wDataHead(axonID, p);
         pvdata_t * patch_dwData = get_dwDataHead(axonID, p);
         for(int k=0; k<nxp*nyp*nfp; k++) {
//            pvdata_t y = postSynapticLayer()->getLayerData(getDelay(axonID))[k];
//            pvdata_t decayterm = 1-y;
//            pvdata_t W = patch_wData[k];
//            if(decayterm <= 0) {
//               decayterm = 0;
//            }
//            else {
//               decayterm *= y;
//               decayterm *= W*(1-W);
//            }
            pvdata_t decayterm = patch_wData[k];
            // decayterm *= 1-decayterm;
            patch_dwData[k] += -weightDecayRate * decayterm + weightNoiseLevel * (2*pv_random_prob()-1);
         }
      }
   }
   return status;
}  // end of GenerativeConn::calc_dW(int);

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

}  // end of namespace PV block
