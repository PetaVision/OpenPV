/*
 * FeedbackConn.cpp
 *
 *  Created on: Nov 15, 2010
 *      Author: pschultz
 */

#include "FeedbackConn.hpp"

namespace PV {

FeedbackConn::FeedbackConn() {
    initialize_base();
}

FeedbackConn::FeedbackConn(const char * name, HyPerCol * hc, KernelConn * ffconn) {
    initialize_base();
    initialize(name, hc, ffconn);
}  // end of FeedbackConn::FeedbackConn(const char *, HyPerCol *, int, GenerativeConn *)

int FeedbackConn::initialize_base() {
   return PV_SUCCESS;
}

int FeedbackConn::initialize(const char * name, HyPerCol *hc, KernelConn * ffconn) {
   TransposeConn::initialize(name, hc, ffconn->postSynapticLayer(), ffconn->preSynapticLayer(), ffconn);
   return PV_SUCCESS;
}

int FeedbackConn::readPatchSize(PVParams * params) {
   int status = PV_SUCCESS;

   int xscaleDiff = pre->getXScale() - post->getXScale();
   // If feedforward conn is many-to-one, feedback conn is one-to-many.
   // Then xscaleDiff > 0.
   // Similarly, if feedforwardConn is one-to-many, xscaleDiff < 0.
   int yscaleDiff = pre->getYScale() - post->getYScale();

   nxp = originalConn->xPatchSize();
   if(xscaleDiff > 0 ) {
       nxp *= (int) powf( 2, xscaleDiff );
   }
   else if(xscaleDiff < 0) {
       nxp /= (int) powf(2,-xscaleDiff);
       assert(originalConn->xPatchSize()==nxp*powf( 2, (float) (-xscaleDiff) ));
   }
   nyp = originalConn->yPatchSize();
   if(yscaleDiff > 0 ) {
       nyp *= (int) powf( 2, (float) yscaleDiff );
   }
   else if(yscaleDiff < 0) {
       nyp /= (int) powf(2,-yscaleDiff);
       assert(originalConn->yPatchSize()==nyp*powf( 2, (float) (-yscaleDiff) ));
   }
   assert( checkPatchSize(nxp, pre->getXScale(), post->getXScale(), 'x') ==
           PV_SUCCESS );
   assert( checkPatchSize(nyp, pre->getYScale(), post->getYScale(), 'y') ==
           PV_SUCCESS );
   return status;
}

int FeedbackConn::readNfp(PVParams * params) {
   nfp = post->getLayerLoc()->nf;
   assert(originalConn && nfp==originalConn->preSynapticLayer()->getLayerLoc()->nf);
   return PV_SUCCESS;
}

int FeedbackConn::setPatchSize(const char * filename) {
   int status = PV_SUCCESS;
   if( filename != NULL ) {
      PVParams * inputParams = parent->parameters();
      bool useListOfArborFiles = inputParams->value(name, "useListOfArborFiles", false)!=0;
      bool combineWeightFiles = inputParams->value(name, "combineWeightFiles", false)!=0;
      if( !useListOfArborFiles && !combineWeightFiles) status = patchSizeFromFile(filename);
   }
   return status;
}  // end of FeedbackConn::setPatchSize(const char *)

PVPatch *** FeedbackConn::initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches,
      const char * filename) {
    if( filename ) return KernelConn::initializeWeights(arbors, dataStart, numPatches, filename);

    transposeKernels();
    return arbors;
}  // end of FeedbackConn::initializeWeights

}  // end of namespace PV block

