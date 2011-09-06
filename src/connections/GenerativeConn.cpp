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
   dWPatches = NULL;
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


    patchindices = (int *) malloc( pre->getNumExtended()*sizeof(int) );
    if( patchindices==NULL ) {
       fprintf(stderr,"GenerativeConn \"%s\": unable to allocate memory for patchindices\n",name);
       exit(EXIT_FAILURE);
    }
    //int axonID = 0; // For now, only one arbor.
    // if we add arbors, patchindices and dWPatches will need to take the arbor index as an argument
    int numKernelPatches = numDataPatches();
    dWPatches = (PVPatch **) malloc(numKernelPatches*sizeof(PVPatch *));
    if( patchindices==NULL ) {
       fprintf(stderr,"GenerativeConn \"%s\": unable to allocate memory for dWPatches\n",name);
       exit(EXIT_FAILURE);
    }
    for( int kex=0; kex<pre->getNumExtended(); kex++ ) {
       patchindices[kex] = this->patchIndexToKernelIndex(kex);
    }
    for( int kernelindex=0; kernelindex<numKernelPatches; kernelindex++ ) {
       dWPatches[kernelindex] = pvpatch_inplace_new(nxp,nyp,nfp);
    }
    // Initializing dWPatch goes here
    return PV_SUCCESS;
}

int GenerativeConn::calc_dW(int axonID) {
   // compute dW but don't add them to the weights yet.
   // That takes place in reduceKernels, so that the output is
   // (in theory) independent of the number of processors.
   int nPre = preSynapticLayer()->getNumNeurons();
   int nx = preSynapticLayer()->getLayerLoc()->nx;
   int ny = preSynapticLayer()->getLayerLoc()->ny;
   int nf = preSynapticLayer()->getLayerLoc()->nf;
   int pad = preSynapticLayer()->getLayerLoc()->nb;
   int numKernelIndices = numDataPatches();
   for( int kernelindex=0; kernelindex<numKernelIndices; kernelindex++ ) {
      int numpatchitems = dWPatches[kernelindex]->nx * dWPatches[kernelindex]->ny * dWPatches[kernelindex]->nf;
      pvdata_t * dwpatchdata = dWPatches[kernelindex]->data;
      for( int n=0; n<numpatchitems; n++ ) {
         dwpatchdata[n] = 0.0;
      }
   }
   for(int kPre=0; kPre<nPre;kPre++) {
      int kExt = kIndexExtended(kPre, nx, ny, nf, pad);
      int kernelindex = patchindices[kExt];
      PVAxonalArbor * arbor = axonalArbor(kPre, axonID);
      size_t offset = arbor->offset;
      pvdata_t preact = preSynapticLayer()->getCLayer()->activity->data[kExt];
      int nyp = arbor->weights->ny;
      int nk = arbor->weights->nx * arbor->weights->nf;
      pvdata_t * postactRef = &(postSynapticLayer()->getCLayer()->activity->data[offset]);
      int sya = arbor->data->sy;
      pvdata_t * dwpatch = dWPatches[kernelindex]->data;
      int syw = dWPatches[kernelindex]->sy;
      for( int y=0; y<nyp; y++ ) {
         int lineoffsetw = 0;
         int lineoffseta = 0;
         for( int k=0; k<nk; k++ ) {
            dwpatch[lineoffsetw + k] += preact*postactRef[lineoffseta + k];
         }
         lineoffsetw += syw;
         lineoffseta += sya;
      }
   }
   // normalizeWeights now called in KernelConn::updateState // normalizeWeights( kernelPatches, numDataPatches(0) );
   lastUpdateTime = parent->simulationTime();

   return PV_SUCCESS;
}  // end of GenerativeConn::calc_dW(int);

#ifdef PV_USE_MPI
int GenerativeConn::reduceKernels(const int axonID) {
   const int numPatches = numDataPatches();
   int idx;
   // Add all the dWPatches from all the processors
   const size_t patchSize = nxp*nyp*nfp*sizeof(pvdata_t);
   const size_t localSize = numPatches * patchSize;

   // Copy this column's dW into mpiReductionBuffer
   idx = 0;
   for (int k = 0; k < numPatches; k++) {
      PVPatch * p = dWPatches[k];
      const pvdata_t * data = p->data;

      const int sxp = p->sx;
      const int syp = p->sy;
      const int sfp = p->sf;

      for (int y = 0; y < p->ny; y++) {
         for (int x = 0; x < p->nx; x++) {
            for (int f = 0; f < p->nf; f++) {
               mpiReductionBuffer[idx] = data[x*sxp + y*syp + f*sfp];
               idx++;
            }
         }
      }
   }

   // MPI_Allreduce combines all processor's buffers and puts the common result
   // into each processor's buffer.
   Communicator * comm = parent->icCommunicator();
   const MPI_Comm mpi_comm = comm->communicator();
   int ierr;
   ierr = MPI_Allreduce(MPI_IN_PLACE, mpiReductionBuffer, localSize, MPI_FLOAT, MPI_SUM, mpi_comm);
   // TODO error handling

   // mpiReductionBuffer now holds the sum over all processes.
   // Don't average; if this were a one-processor task the change to one patch in W would be the sum
   // of all the changes from each presynaptic neuron pointing to that W-patch.
   idx = 0;
   for (int k = 0; k < numPatches; k++) {
      PVPatch * p = dWPatches[k];
      pvdata_t * data = p->data;

      const int sxp = p->sx;
      const int syp = p->sy;
      const int sfp = p->sf;

      for (int y = 0; y < p->ny; y++) {
         for (int x = 0; x < p->nx; x++) {
            for (int f = 0; f < p->nf; f++) {
               data[x*sxp + y*syp + f*sfp] = mpiReductionBuffer[idx];
               idx++;
            }
         }
      }
   }

   return PV_SUCCESS;
}  // end of GenerativeConn::reduceKernels(int)
#endif // PV_USE_MPI

int GenerativeConn::updateWeights(int axonID) {
   const int numPatches = numDataPatches();
   for( int k=0; k<numPatches; k++ ) {
      PVPatch * w = getKernelPatch(axonID, k);
      pvdata_t * wdata = w->data;
      PVPatch * dw = dWPatches[k];
      const int sxp = w->sx;
      const int syp = w->sy;
      const int sfp = w->sf;
      for( int y = 0; y < w->ny; y++ ) {
         for( int x = 0; x < w->nx; x++ ) {
            for( int f = 0; f < w->nf; f++ ) {
               int idx = f*sfp + x*sxp + y*syp;
               wdata[idx] += relaxation*dw->data[idx];
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

PVPatch ** GenerativeConn::normalizeWeights(PVPatch ** patches, int numPatches, int arborId) {
   int neuronsperpatch;
   switch( normalizeMethod ) {
   case 0:
      break;
   case 1:
      patches = KernelConn::normalizeWeights(patches, numPatches, arborId);
      break;
   case 2:
      neuronsperpatch = (patches[0]->nx)*(patches[0]->ny)*(patches[0]->nf);
      for( int n=0; n<neuronsperpatch; n++ ) {
         pvdata_t s = 0;
         for( int k=0; k<numPatches; k++ ) {
            pvdata_t d = patches[k]->data[n];
            s += d*d;
         }
         for( int k=0; k<numPatches; k++ ) {
            patches[k]->data[n] *= normalizeConstant/sqrt(s);
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
         for( int n=0; n<neuronsperpatch; n++ ) {
            curpatch->data[n] *= normalizeConstant/sqrt(s);
         }
      }
      break;
   default:
      assert(false); // This possibility was eliminated in initNormalize().
      break;
   }
   return patches;
}  // end of GenerativeConn::normalizeWeights

}  // end of namespace PV block
