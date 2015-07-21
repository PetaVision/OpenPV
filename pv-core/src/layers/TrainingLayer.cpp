/*
 * TrainingLayer.cpp
 *
 *  Created on: Dec 8, 2010
 *      Author: pschultz
 */

#include "TrainingLayer.hpp"

namespace PV {

TrainingLayer::TrainingLayer() {
   initialize_base();
}

TrainingLayer::TrainingLayer(const char * name, HyPerCol * hc) {
   initialize_base();
   initialize(name, hc);
}

TrainingLayer::~TrainingLayer() {
   free(filename);
   free(trainingLabels);
}

int TrainingLayer::initialize_base() {
   numChannels = 0;
   trainingLabels = NULL;
   return PV_SUCCESS;
}

int TrainingLayer::initialize(const char * name, HyPerCol * hc) {
   int status = ANNLayer::initialize(name, hc);

   curTrainingLabelIndex = 0;
   PVParams * params = parent->parameters();
   assert(!parent->parameters()->presentAndNotBeenRead(name, "displayPeriod"));
   assert(!parent->parameters()->presentAndNotBeenRead(name, "distToData"));
   if( displayPeriod < 0 || distToData < 0) {
      fprintf(stderr, "Constructor for TrainingLayer \"%s\" requires parameters displayPeriod and distToData to be set to nonnegative values in the params file.\n", name);
      exit(PV_FAILURE);
   }
   nextUpdateTime = displayPeriod + distToData;
   return status;
}

int TrainingLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_trainingLabelsPath(ioFlag);
   ioParam_displayPeriod(ioFlag);
   ioParam_distToData(ioFlag);
   ioParam_strength(ioFlag);
   return PV_SUCCESS;
}
void TrainingLayer::ioParam_trainingLabelsPath(enum ParamsIOFlag ioFlag) {
   parent->ioParamString(ioFlag, name, "trainingLabelsPath", &filename, NULL, true/*warnIfAbsent*/);
}

void TrainingLayer::ioParam_displayPeriod(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "displayPeriod", &displayPeriod, -1.0, true/*warnIfAbsent*/);
}

void TrainingLayer::ioParam_distToData(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "distToData", &distToData, -1.0, true/*warnIfAbsent*/);
}

void TrainingLayer::ioParam_strength(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "strength", &strength, (pvdata_t) 1, true/*warnIfAbsent*/);
}

int TrainingLayer::allocateDataStructures() {
   ANNLayer::allocateDataStructures();
   errno = 0;
   numTrainingLabels = readTrainingLabels( filename, &this->trainingLabels ); // trainingLabelsFromFile allocated within this readTrainingLabels call
   if( this->trainingLabels == NULL) return PV_FAILURE;
   if( numTrainingLabels <= 0) {
      fprintf(stderr, "Training Layer \"%s\": No training labels.  Exiting\n", name);
      exit( errno ? errno : EXIT_FAILURE );
   }
   return PV_SUCCESS;
}

int TrainingLayer::readTrainingLabels(const char * filename, int ** trainingLabelsFromFile) {
   PV_Stream * instream = PV_fopen(filename, "r", false/*verifyWrites*/);
   if( instream == NULL ) {
      fprintf( stderr, "TrainingLayer error opening \"%s\": %s\n", filename, strerror(errno) );
      *trainingLabelsFromFile = NULL;
      return 0;
   }

   int didReadLabel;
   int n = 0;
   int label;
   int * labels = NULL;
   int * oldlabels;
   do {
      didReadLabel = fscanf(instream->fp, "%d", &label);
      updatePV_StreamFilepos(instream); // to recalculate instream->filepos since it's not easy to tell how many characters were read
      switch( didReadLabel ) {
      case 0:
         PV_fseek( instream, 1L, SEEK_CUR );
         break;
      case 1:
         n++;
         oldlabels = labels;
         labels = (int *) malloc((size_t) n * sizeof(int) );
         assert(labels);
         for(int k=0; k<n-1; k++) labels[k] = oldlabels[k];
         labels[n-1] = label;
         free(oldlabels);
         break;
      }
   } while( didReadLabel != EOF );
   PV_fclose(instream);
   *trainingLabelsFromFile = labels;
   return n;
}

int TrainingLayer::initializeV() {
   memset(getV(), 0, ((size_t) getNumNeurons())*sizeof(pvdata_t));
   // above line not necessary if V was allocated with calloc
   getV()[trainingLabels[curTrainingLabelIndex]] = strength;
   return PV_SUCCESS;
}

bool TrainingLayer::needUpdate(double timed, double dt) {
   return timed >= nextUpdateTime;
}

int TrainingLayer::updateState(double timed, double dt) {
   int status = updateState(timed, dt, getLayerLoc(), getCLayer()->activity->data, getV(), numTrainingLabels, trainingLabels, curTrainingLabelIndex, strength);
   return status;
}

double TrainingLayer::getDeltaUpdateTime() {
   return displayPeriod;
}

int TrainingLayer::updateState(double timed, double dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int numTrainingLabels, int * trainingLabels, int traininglabelindex, int strength) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int status = updateV_TrainingLayer(num_neurons, V, numTrainingLabels, trainingLabels, curTrainingLabelIndex, strength);
   assert(status == PV_SUCCESS);
   curTrainingLabelIndex++;
   setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
   return PV_SUCCESS;
}

int TrainingLayer::readStateFromCheckpoint(const char * cpDir, double * timeptr) {
   int status = ANNLayer::readStateFromCheckpoint(cpDir, timeptr);
   status = read_currentLabelIndexFromCheckpoint(cpDir);
   return status;
}

int TrainingLayer::read_currentLabelIndexFromCheckpoint(const char * cpDir) {
   return parent->readScalarFromFile(cpDir, getName(), "currentLabelIndex", &curTrainingLabelIndex, curTrainingLabelIndex);
}

int TrainingLayer::checkpointWrite(const char * cpDir) {
   int status = HyPerLayer::checkpointWrite(cpDir);
   assert(status == PV_SUCCESS);
   parent->writeScalarToFile(cpDir, getName(), "currentLabelIndex", curTrainingLabelIndex);
   return status;
}

} // end of namespace PV
