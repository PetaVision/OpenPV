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
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel) {
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL, NULL);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
//   HyPerLayer *, HyPerLayer *, int)
GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      InitWeights *weightInit) {
   initialize_base();
   initialize(name, hc, pre, post, channel, NULL, weightInit);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
//   HyPerLayer *, HyPerLayer *, int)

GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      const char * filename) {
   initialize_base();
   initialize(name, hc, pre, post, channel, filename, NULL);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
//   HyPerLayer *, HyPerLayer *, int, const char *)
GenerativeConn::GenerativeConn(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      const char * filename, InitWeights *weightInit) {
   initialize_base();
   initialize(name, hc, pre, post, channel, filename, weightInit);
}  // end of GenerativeConn::GenerativeConn(const char *, HyPerCol *,
//   HyPerLayer *, HyPerLayer *, int, const char *)

int GenerativeConn::initialize_base() {
   plasticityFlag = true; // Default value; override in params
   weightUpdatePeriod = 1;   // Default value; override in params

   relaxation = 1.0;
   nonnegConstraintFlag = false;
   normalizeMethod = 0;
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
      HyPerLayer * pre, HyPerLayer * post, ChannelType channel,
      const char * filename, InitWeights *weightInit) {
   PVParams * params = hc->parameters();
   relaxation = params->value(name, "relaxation", 1.0f);
   nonnegConstraintFlag = (bool) params->value(name, "nonnegConstraintFlag", 0.f); // default is not to constrain nonnegative.
   KernelConn::initialize(name, hc, pre, post, channel, filename, weightInit);

   //GenerativeConn has not been updated to support multiple arbors!
   assert(numberOfAxonalArborLists()==1);
   // For now, only one arbor.
   // If we add arbors, patchindices will need to take the arbor index as an argument

   patchindices = (int *) malloc( pre->getNumExtended()*sizeof(int) );
   if( patchindices==NULL ) {
      fprintf(stderr,"GenerativeConn \"%s\": unable to allocate memory for patchindices\n",name);
      exit(EXIT_FAILURE);
   }
   for( int kex=0; kex<pre->getNumExtended(); kex++ ) {
      patchindices[kex] = this->patchIndexToKernelIndex(kex);
   }
   return PV_SUCCESS;
}

int GenerativeConn::update_dW(int axonID) {
   return defaultUpdate_dW(axonID);
}  // end of GenerativeConn::calc_dW(int);

int GenerativeConn::updateWeights(int axonID) {
   const int numPatches = numDataPatches();
   for( int k=0; k<numPatches; k++ ) {
      PVPatch * w = getKernelPatch(axonID, k);
      pvdata_t * wdata = w->data;
      PVPatch * dw = dKernelPatches[0][k];
      pvdata_t * dwdata = dw->data;
      const int sxp = w->sx;
      const int syp = w->sy;
      const int sfp = w->sf;
      for( int y = 0; y < w->ny; y++ ) {
         for( int x = 0; x < w->nx; x++ ) {
            for( int f = 0; f < w->nf; f++ ) {
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
      normalizeConstant = params->value(name, "normalizeConstant", 1.0f);
      break;
   default:
      fprintf(stderr,"Connection \"%s\": Unrecognized normalizeMethod %d.  Using normalizeMethod = 1 (KernelConn's normalizeWeights).\n", this->getName(), normalizeMethod);
      KernelConn::initNormalize();
      break;
   }
   return PV_SUCCESS;
}

int GenerativeConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId) {
   int status = PV_SUCCESS;
   int neuronsperpatch;
   switch( normalizeMethod ) {
   case 0:
      break;
   case 1:
      status = KernelConn::normalizeWeights(patches, numPatches, arborId);
      break;
   case 2:
      neuronsperpatch = (patches[0]->nx)*(patches[0]->ny)*(patches[0]->nf);
      for( int n=0; n<neuronsperpatch; n++ ) {
         pvdata_t s = 0;
         for( int k=0; k<numPatches; k++ ) {
            pvdata_t d = patches[k]->data[n];
            s += d*d;
         }
         s = sqrt(s);
         for( int k=0; k<numPatches; k++ ) {
            patches[k]->data[n] *= normalizeConstant/s;
         }
      }
      break;
   case 3:
      neuronsperpatch = (patches[0]->nx)*(patches[0]->ny)*(patches[0]->nf);
      for( int k=0; k<numPatches; k++ ) {
         PVPatch * curpatch = patches[k];
         pvdata_t s = 0;
         for( int n=0; n<neuronsperpatch; n++ ) {
            pvdata_t d = curpatch->data[n];
            s += d*d;
         }
         s = sqrt(s);
         for( int n=0; n<neuronsperpatch; n++ ) {
            curpatch->data[n] *= normalizeConstant/s;
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
