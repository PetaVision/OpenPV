/*
 * IncrementLayer.cpp
 *
 *  Created on: Feb 7, 2012
 *      Author: pschultz
 */

#include "IncrementLayer.hpp"
#include "updateStateFunctions.h"

namespace PV {

IncrementLayer::IncrementLayer() {
   initialize_base();
}

IncrementLayer::IncrementLayer(const char* name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

IncrementLayer::~IncrementLayer() {
   free(Vprev);
}

int IncrementLayer::initialize_base() {
   Vprev = NULL;
   displayPeriod = 0;
   VInited = false;
   nextUpdateTime = 0;
   return PV_SUCCESS;
}

int IncrementLayer::initialize(const char* name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);
   nextUpdateTime = firstUpdateTime+displayPeriod;

   // Why isn't this in allocateDataStructures()?
   // It is!  See the call to allocateBuffer()
   // Vprev = (pvdata_t *) calloc(getNumNeuronsAllBatches(),sizeof(pvdata_t));
   // if( Vprev == NULL ) {
   //    fprintf(stderr, "Unable to allocate Vprev buffer for IncrementLayer \"%s\"\n", name);
   //    abort();
   // }
   // for( int k=0; k<getNumNeurons(); k++ ) {
   //    assert(GSyn[0][k]==0 && GSyn[1][k]==0);
   // }

   return status;
}

int IncrementLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_displayPeriod(ioFlag);
   ioParam_firstUpdateTime(ioFlag);
   return status;
}

void IncrementLayer::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "displayPeriod", &displayPeriod, parent->getDeltaTime());
}

void IncrementLayer::ioParam_firstUpdateTime(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "firstUpdateTime", &firstUpdateTime, parent->simulationTime());
}

// Threshold paramaters are not used, as updateState does not call applyVMax or applyVThresh
// Override relevant ANNLayer::ioParam_* methods so that the threshold params are not read
// from the params file, thereby creating an unnecessary warning.

void IncrementLayer::ioParam_VThresh(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) VThresh = max_pvvdata_t;
}

void IncrementLayer::ioParam_AMin(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) AMin = -max_pvadata_t;
}

void IncrementLayer::ioParam_AMax(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) AMax = -max_pvadata_t;
}

void IncrementLayer::ioParam_AShift(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) AShift = (pvdata_t) 0;

}

void IncrementLayer::ioParam_VWidth(enum ParamsIOFlag ioFlag) {
   if (ioFlag==PARAMS_IO_READ) VWidth = (pvdata_t) 0;
}

int IncrementLayer::setVertices() {
   slopeNegInf = 1.0f;
   slopePosInf = 0.0f;
   numVertices = 1;
   verticesV = (pvpotentialdata_t *) calloc((size_t) numVertices, sizeof(*verticesV));
   verticesA = (pvadata_t *) calloc((size_t) numVertices, sizeof(*verticesA));
   if (verticesV==NULL || verticesA==NULL) {
      fprintf(stderr, "%s \"%s\" error: unable to allocate memory for vertices:%s\n",
            parent->parameters()->groupKeywordFromName(name), name, strerror(errno));
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int IncrementLayer::allocateDataStructures() {
   int status = ANNLayer::allocateDataStructures();

   if (status == PV_SUCCESS) status = allocateBuffer(&Vprev, getNumNeurons(), "V at previous time");
   if (status == PV_SUCCESS) {
      assert(GSyn && GSyn[0] && GSyn[1]);
      for( int k=0; k<getNumNeurons(); k++ ) {
         assert(GSyn[0][k]==0 && GSyn[1][k]==0);
      }
   }
   if (status != PV_SUCCESS) exit(EXIT_FAILURE);

   return status;
}

int IncrementLayer::updateState(double timef, double dt) {
   int status;
   assert(numChannels>=2);  // updateState uses gSynExc and gSynInh
   status = doUpdateState(timef, dt, &VInited, &nextUpdateTime, firstUpdateTime, displayPeriod, getLayerLoc(), getCLayer()->activity->data, getV(), getVprev(), getNumChannels(), GSyn[0]);
   return status;
}

int IncrementLayer::doUpdateState(double timef, double dt, bool * inited, double * next_update_time, double first_update_time, double display_period, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, pvdata_t * Vprev, int num_channels, pvdata_t * gSynHead) {
   int status = PV_SUCCESS;
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int nbatch = loc->nbatch;

   if( *inited ) {
      if( timef >= *next_update_time ) {
         *next_update_time += display_period;
         pvdata_t * Vprev1 = Vprev;
         pvdata_t * V = getV();
         for( int k=0; k<num_neurons; k++ ) {
            *(Vprev1++) = *(V++);
         }
      }
      status = applyGSyn_HyPerLayer(num_neurons, V, gSynHead);
      if( status == PV_SUCCESS ) status = setActivity_IncrementLayer(num_neurons, A, V, Vprev, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up); // setActivity();
   }
   else {
      if( timef >= first_update_time ) {
         status = updateV_PtwiseLinearTransferLayer(num_neurons, V, num_channels, gSynHead, A, numVertices, verticesV, verticesA, slopes, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up); // updateV();
         resetGSynBuffers_HyPerLayer(num_neurons, num_channels, gSynHead);
         *inited = true;
      }
   }
   return status;

}

int IncrementLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   return setActivity_IncrementLayer(getNumNeurons(), clayer->activity->data, getV(), getVprev(), loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
}

int IncrementLayer::readVprevFromCheckpoint(const char * cpDir, double * timeptr) {
   char * filename = parent->pathInCheckpoint(cpDir, getName(), "_Vprev.pvp");
   int status = readBufferFile(filename, parent->icCommunicator(), timeptr, &Vprev, 1, /*extended*/false, getLayerLoc());
   assert(status==PV_SUCCESS);
   return status;
}

int IncrementLayer::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = ANNLayer::readStateFromCheckpoint(cpDir, timeptr);
   status = readVprevFromCheckpoint(cpDir, timeptr);
   // TODO: check whether VInited should be set if initializeFromCheckpointFlag is true
   return status;
}

int IncrementLayer::checkpointRead(const char * cpDir, double * timeptr) {
   int status = HyPerLayer::checkpointRead(cpDir, timeptr);
   VInited = *timeptr >= firstUpdateTime + parent->getDeltaTime();

   return PV_SUCCESS;
}

int IncrementLayer::checkpointWrite(const char * cpDir) {
   HyPerLayer::checkpointWrite(cpDir);
   InterColComm * icComm = parent->icCommunicator();
   double timed = (double) parent->simulationTime();
   int filenamesize = strlen(cpDir)+1+strlen(name)+12;
   // The +1 is for the slash between cpDir and name; the +12 needs to be large enough to hold the suffix (e.g. _G_Gap.pvp) plus the null terminator
   char * filename = (char *) malloc( filenamesize*sizeof(char) );
   assert(filename != NULL);
   int chars_needed = snprintf(filename, filenamesize, "%s/%s_Vprev.pvp", cpDir, name);
   assert(chars_needed < filenamesize);
   writeBufferFile(filename, icComm, timed, &Vprev, 1, /*extended*/false, getLayerLoc());
   free(filename);
   return PV_SUCCESS;
}


} /* namespace PV */
