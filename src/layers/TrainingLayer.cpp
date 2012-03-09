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

TrainingLayer::TrainingLayer(const char * name, HyPerCol * hc, const char * filename) {
   initialize_base();
   initialize(name, hc, filename);
}
/*
TrainingLayer::TrainingLayer(const char * name, HyPerCol * hc, const char * filename)
: ANNLayer(name, hc) {
   initialize( filename, hc->parameters() );
}
*/

TrainingLayer::~TrainingLayer() {
   free(trainingLabels);
}

int TrainingLayer::initialize_base() {
   trainingLabels = NULL;
   return PV_SUCCESS;
}

int TrainingLayer::initialize(const char * name, HyPerCol * hc, const char * filename) {
   int status = ANNLayer::initialize(name, hc, MAX_CHANNELS);
   PVParams * params = hc->parameters();
   float displayPeriod = params->value(name, "displayPeriod", -1);
   float distToData = params->value(name, "distToData", -1);
   strength = params->value(name, "strength", 1);
   if( displayPeriod < 0 || distToData < 0) {
      fprintf(stderr, "Constructor for TrainingLayer \"%s\" requires parameters displayPeriod and distToData to be set to nonnegative values in the params file.\n", name);
      exit(PV_FAILURE);
   }
   errno = 0;
   numTrainingLabels = readTrainingLabels( filename, &this->trainingLabels ); // trainingLabelsFromFile allocated within this readTrainingLabels call
   if( this->trainingLabels == NULL) return PV_FAILURE;
   if( numTrainingLabels <= 0) {
      fprintf(stderr, "Training Layer \"%s\": No training labels.  Exiting\n", name);
      exit( errno ? errno : EXIT_FAILURE );
   }
   curTrainingLabelIndex = 0;
   this->displayPeriod = displayPeriod;
   this->distToData = distToData;
   this->nextLabelTime = displayPeriod + distToData;

   pvdata_t * V = getV();
   for( int k=0; k < getNumNeurons(); k++ ) V[k] = 0;
   // above line not necessary if V was allocated with calloc
   getV()[trainingLabels[curTrainingLabelIndex]] = strength; // setLabeledNeuron();
   const PVLayerLoc * loc = getLayerLoc();
   setActivity_HyPerLayer(getNumNeurons(), getCLayer()->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->nb);
   // needed because updateState won't call setActivity until the first update period has passed.
   // setActivity();
   return status;
}

int TrainingLayer::readTrainingLabels(const char * filename, int ** trainingLabelsFromFile) {
   FILE * instream = fopen(filename, "r");
   if( instream == NULL ) {
      fprintf( stderr, "TrainingLayer: Unable to open \"%s\". Error %d\n", filename, errno );
      *trainingLabelsFromFile = NULL;
      return 0;
   }

   int didReadLabel;
   int n = 0;
   int label;
   int * labels = NULL;
   int * oldlabels;
   do {
      didReadLabel = fscanf(instream, "%d", &label);
      switch( didReadLabel ) {
      case 0:
         fseek( instream, 1L, SEEK_CUR );
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
   fclose(instream);
   *trainingLabelsFromFile = labels;
   return n;
}

int TrainingLayer::updateState(float timef, float dt) {
   int status = PV_SUCCESS;
   if(timef >= nextLabelTime) {
      nextLabelTime += displayPeriod;
      status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), numTrainingLabels, trainingLabels, curTrainingLabelIndex, strength);
   }
   return status;
}

int TrainingLayer::updateState(float timef, float dt, const PVLayerLoc * loc, pvdata_t * A, pvdata_t * V, int numTrainingLabels, int * trainingLabels, int traininglabelindex, int strength) {
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   updateV_TrainingLayer(num_neurons, V, numTrainingLabels, trainingLabels, curTrainingLabelIndex, strength);
   curTrainingLabelIndex++;
   setActivity_HyPerLayer(num_neurons, A, V, nx, ny, nf, loc->nb); // setActivity();
   // resetGSynBuffers(); // Since V doesn't use GSyn when updating, no need to reset GSyn buffers
   updateActiveIndices();
      // return ANNLayer::updateState(time, dt);
   return PV_SUCCESS;
}

//int TrainingLayer::updateV() {
//   int status1 = clearLabeledNeuron();
//
//   curTrainingLabelIndex++;
//   curTrainingLabelIndex = curTrainingLabelIndex == numTrainingLabels ? 0 : curTrainingLabelIndex;
//   int status2 = setLabeledNeuron();
//   return (status1==PV_SUCCESS && status2==PV_SUCCESS) ?
//         PV_SUCCESS : PV_FAILURE;
//}  // end of TrainingLayer::updateV()

//int TrainingLayer::setLabeledNeuronToValue(pvdata_t val) {
//   int n = trainingLabels[curTrainingLabelIndex];
//   int N = getNumNeurons();
//   if( n>=N ) {
//      sendBadNeuronMessage();
//      return PV_FAILURE;
//   }
//   else {
//      pvdata_t * V = getV();
//      V[trainingLabels[curTrainingLabelIndex]] = val;
//      return PV_SUCCESS;
//   }
//}  // end of TrainingLayer::setLabeledNeuronToValue(int, pvdata_t)

//void TrainingLayer::sendBadNeuronMessage() {
//   fprintf(stderr, "TrainingLayer \"%s\":\n", name);
//   fprintf(stderr, "Number of training labels is %d\n", numTrainingLabels);
//   fprintf(stderr, "Current label index is %d\n", curTrainingLabelIndex);
//   fprintf(stderr, "Value of label %d is %d\n", curTrainingLabelIndex,
//         trainingLabels[curTrainingLabelIndex]);
//   fprintf(stderr, "Number of neurons is %d\n", getNumNeurons());
//}

}  // end of namespace PV
