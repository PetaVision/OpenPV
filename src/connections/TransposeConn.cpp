/*
 * TransposeConn.cpp
 *
 *  Created on: May 16, 2011
 *      Author: peteschultz
 */

#include "TransposeConn.hpp"

namespace PV {

TransposeConn::TransposeConn() {
    initialize_base();
}  // TransposeConn::~TransposeConn()

TransposeConn::TransposeConn(const char * name, HyPerCol * hc, HyPerLayer * preLayer, HyPerLayer * postLayer, KernelConn * auxConn) {
    initialize_base();
    initialize(name, hc, preLayer, postLayer, auxConn);
}  // TransposeConn::TransposeConn(const char * name, HyPerCol * hc, HyPerLayer *, HyPerLayer *, ChannelType, KernelConn *)

TransposeConn::~TransposeConn() {

}  // TransposeConn::~TransposeConn()

int TransposeConn::initialize_base() {
   plasticityFlag = true; // Default value; override in params
   weightUpdatePeriod = 1;   // Default value; override in params
   // TransposeConn::initialize_base() gets called after
   // KernelConn::initialize_base() so these default values override
   // those in KernelConn::initialize_base().
   // TransposeConn::initialize_base() gets called before
   // KernelConn::initialize(), so these values still get overridden
   // by the params file values.

   originalConn = NULL;
   return PV_SUCCESS;
}  // TransposeConn::initialize_base()

int TransposeConn::initialize(const char * name, HyPerCol * hc, HyPerLayer * preLayer, HyPerLayer * postLayer, KernelConn * auxConn) {

   originalConn = auxConn;
   normalize_flag = false; // HyPerConn::initializeWeights never gets called (most of what it does isn't needed) so initNormalize never gets called
   int status = KernelConn::initialize(name, hc, preLayer, postLayer, NULL, NULL);
   assert(numberOfAxonalArborLists()==1);
   return status;
}

void TransposeConn::readNumAxonalArbors(PVParams * params) {
   assert(originalConn);
   numAxonalArborLists = originalConn->numberOfAxonalArborLists();
   //TransposeConn has not been updated to support multiple arbors
   if (numAxonalArborLists!=1) {
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposeConn error for connection \"%s\": original conn \"%s\" must have only one arbor.\n", name, originalConn->getName());
      }
      MPI_Barrier(getParent()->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
}

int TransposeConn::readPatchSize(PVParams * params) {
   // If originalConn is many-to-one, the transpose connection is one-to-many.
   // Then xscaleDiff > 0.
   // Similarly, if originalConn is one-to-many, xscaleDiff < 0.
   int xscaleDiff = pre->getXScale() - post->getXScale();
   nxp = originalConn->xPatchSize();
   if(xscaleDiff > 0 ) {
       nxp *= (int) pow( 2, xscaleDiff );
   }
   else if(xscaleDiff < 0) {
       nxp /= (int) pow(2,-xscaleDiff);
       assert(originalConn->xPatchSize()==nxp*pow( 2, (float) (-xscaleDiff) ));
   }
   int yscaleDiff = pre->getYScale() - post->getYScale();
   nyp = originalConn->yPatchSize();
   if(yscaleDiff > 0 ) {
       nyp *= (int) pow( 2, (float) yscaleDiff );
   }
   else if(yscaleDiff < 0) {
       nyp /= (int) pow(2,-yscaleDiff);
       assert(originalConn->yPatchSize()==nyp*pow( 2, (float) (-yscaleDiff) ));
   }
   assert( checkPatchSize(nxp, pre->getXScale(), post->getXScale(), 'x') ==
           PV_SUCCESS );
   assert( checkPatchSize(nyp, pre->getYScale(), post->getYScale(), 'y') ==
           PV_SUCCESS );

   nxpShrunken = nxp;
   nypShrunken = nyp;
   return PV_SUCCESS;
}

int TransposeConn::readNfp(PVParams * params) {
   nfp = post->getLayerLoc()->nf;
   assert(originalConn && nfp==originalConn->preSynapticLayer()->getLayerLoc()->nf);
   return PV_SUCCESS;
}

void TransposeConn::readPlasticityFlag(PVParams * params) {
   assert(originalConn);
   plasticityFlag = originalConn->getPlasticityFlag();
}

void TransposeConn::readCombine_dW_with_W_flag(PVParams * params) {
   combine_dW_with_W_flag = false;
}

void TransposeConn::read_dWMax(PVParams * params) {
   dWMax = 1.0;
}

void TransposeConn::readKeepKernelsSynchronized(PVParams * params) {
   keepKernelsSynchronized_flag = false;
}

void TransposeConn::readWeightUpdatePeriod(PVParams * params) {
   weightUpdatePeriod = 0.0f;  // Ensures that every timestep updateState calls updateWeights, which will compare lastUpdateTime to originalConn's lastUpdateTime
}

void TransposeConn::readInitialWeightUpdateTime(PVParams * params) {
   weightUpdateTime = parent->simulationTime();
}

void TransposeConn::readShrinkPatches(PVParams * params) {
   if(originalConn->getShrinkPatches_flag()) {
      if (parent->columnId()==0) {
         fprintf(stderr, "TransposeConn \"%s\" error: original conn \"%s\" has shrinkPatches set to true.  TransposeConn has not been implemented for that case.\n", name, originalConn->getName());
      }
   }
   shrinkPatches_flag = false;
}

InitWeights * TransposeConn::handleMissingInitWeights(PVParams * params) {
   // TransposeConn doesn't use InitWeights; it initializes the weight by transposing the initial weights of originalConn
   return NULL;
}

PVPatch *** TransposeConn::initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches, const char * filename) {
    if( filename ) {
       return KernelConn::initializeWeights(arbors, dataStart, numPatches, filename);
    }
    else {
        transposeKernels();
    }
    return arbors;
}  // TransposeConn::initializeWeights(PVPatch **, int, const char *)

// int TransposeConn::initNormalize() {
//    normalize_flag = false;
//    return PV_SUCCESS;
// }

int TransposeConn::setPatchSize(const char * filename) {
    int status = PV_SUCCESS;

    assert( checkPatchSize(nyp, pre->getXScale(), post->getXScale(), 'x') ==
            PV_SUCCESS );

    assert( checkPatchSize(nyp, pre->getYScale(), post->getYScale(), 'y') ==
            PV_SUCCESS );

    status = filename ? patchSizeFromFile(filename) : PV_SUCCESS;
    return status;
}  // TransposeConn::setPatchSize(const char *)

int TransposeConn::updateWeights(int axonID) {
   int status;
   float original_update_time = originalConn->getLastUpdateTime();
   if(original_update_time > lastUpdateTime ) {
      status = transposeKernels();
      lastUpdateTime = parent->simulationTime();
   }
   else
      status = PV_SUCCESS;
   return status;
}  // end of TransposeConn::updateWeights(int);

// TODO reorganize transposeKernels():  Loop over kernelNumberFB on outside, call transposeOneKernel(kpFB, kernelnumberFB), which handles all the cases.
// This would play better with Kris's initWeightsMethod.
int TransposeConn::transposeKernels() {
    // compute the transpose of originalConn->kernelPatches and
    // store into this->kernelPatches
    // assume scale factors are 1 and that nxp, nyp are odd.

    int xscalediff = pre->getXScale()-post->getXScale();
    int yscalediff = pre->getYScale()-post->getYScale();
    // scalediff>0 means TransposeConn's post--that is, the originalConn's pre--has a higher neuron density

    int numFBKernelPatches = getNumDataPatches();
    int numFFKernelPatches = originalConn->getNumDataPatches();

    if( xscalediff <= 0 && yscalediff <= 0) {
        int xscaleq = (int) pow(2,-xscalediff);
        int yscaleq = (int) pow(2,-yscalediff);

        for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
            // PVPatch * kpFB = getKernelPatch(0, kernelnumberFB);
            pvdata_t * dataStartFB = get_wDataHead(0, kernelnumberFB);
            int nfFB = nfp;
            assert(numFFKernelPatches == nfFB);
            int nxFB = nxp; // kpFB->nx;
            int nyFB = nyp; // kpFB->ny;
            for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
                for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
                    for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                        int kIndexFB = kIndex(kxFB,kyFB,kfFB,nxFB,nyFB,nfFB);
                        int kernelnumberFF = kfFB;
                        // PVPatch * kpFF = originalConn->getKernelPatch(0, kernelnumberFF);
                        pvdata_t * dataStartFF = originalConn->get_wDataHead(0, kernelnumberFF);
                        int nxpFF = originalConn->xPatchSize();
                        int nypFF = originalConn->yPatchSize();
                        assert(numFBKernelPatches == originalConn->fPatchSize() * xscaleq * yscaleq);
                        int kfFF = featureIndex(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
                        int kxFFoffset = kxPos(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
                        int kxFF = (nxp - 1 - kxFB) * xscaleq + kxFFoffset;
                        int kyFFoffset = kyPos(kernelnumberFB, xscaleq, yscaleq, originalConn->fPatchSize());
                        int kyFF = (nyp - 1 - kyFB) * yscaleq + kyFFoffset;
                        int kIndexFF = kIndex(kxFF, kyFF, kfFF, nxpFF, nypFF, originalConn->fPatchSize());
                        // can the calls to kxPos, kyPos, featureIndex be replaced by one call to patchIndexToKernelIndex?
                        dataStartFB[kIndexFB] = dataStartFF[kIndexFF];
                        // kpFB->data[kIndexFB] = kpFF->data[kIndexFF];
                    }
                }
            }
        }
    }
    else if( xscalediff > 0 && yscalediff > 0) {
        int xscaleq = (int) pow(2,xscalediff);
        int yscaleq = (int) pow(2,yscalediff);
        for( int kernelnumberFB = 0; kernelnumberFB < numFBKernelPatches; kernelnumberFB++ ) {
            // PVPatch * kpFB = getKernelPatch(0, kernelnumberFB);
            pvdata_t * dataStartFB = get_wDataHead(0, kernelnumberFB);
            int nxFB = nxp; // kpFB->nx;
            int nyFB = nyp; // kpFB->ny;
            int nfFB = nfp;
            for( int kyFB = 0; kyFB < nyFB; kyFB++ ) {
                int precelloffsety = kyFB % yscaleq;
                   for( int kxFB = 0; kxFB < nxFB; kxFB++ ) {
                    int precelloffsetx = kxFB % xscaleq;
                    for( int kfFB = 0; kfFB < nfFB; kfFB++ ) {
                        int kernelnumberFF = (precelloffsety*xscaleq + precelloffsetx)*nfFB + kfFB;
                        pvdata_t * dataStartFF = originalConn->get_wDataHead(0, kernelnumberFF);
                        int nxpFF = originalConn->xPatchSize();
                        int nypFF = originalConn->yPatchSize();
                        int kxFF = (nxp-kxFB-1)/xscaleq;
                        assert(kxFF >= 0 && kxFF < originalConn->xPatchSize());
                        int kyFF = (nyp-kyFB-1)/yscaleq;
                        assert(kyFF >= 0 && kyFF < originalConn->yPatchSize());
                        int kfFF = kernelnumberFB;
                        assert(kfFF >= 0 && kfFF < originalConn->fPatchSize());
                        int kIndexFF = kIndex(kxFF, kyFF, kfFF, nxpFF, nypFF, originalConn->fPatchSize());
                        int kIndexFB = kIndex(kxFB, kyFB, kfFB, nxFB, nyFB, nfFB);
                        dataStartFB[kIndexFB] = dataStartFF[kIndexFF];
                    }
                }
            }
        }
    }
    else {
        fprintf(stderr,"xscalediff = %d, yscalediff = %d: the case of many-to-one in one dimension and one-to-many in the other"
                       "has not yet been implemented.\n", xscalediff, yscalediff);
        exit(1);
    }

    return PV_SUCCESS;
}  // TransposeConn::transposeKernels()

int TransposeConn::reduceKernels(int arborID) {
   // Values are taken from originalConn.  If originalConn keeps kernels synchronized, then TransposeConn stays synchronized automatically.
   // If originalConn does not, then TransposeConn shouldn't either.
   return PV_SUCCESS;
}

} // end namespace PV
