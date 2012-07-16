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

int TrainingLayer::initializeState() {
   int status = PV_SUCCESS;
   PVParams * params = parent->parameters();
   bool restart_flag = params->value(name, "restart", 0.0f) != 0.0f;
   if (restart_flag) {
      float timef;
      status = readState(&timef);
   }
   else {
      pvdata_t * V = getV();
      for( int k=0; k < getNumNeurons(); k++ ) V[k] = 0;
      // above line not necessary if V was allocated with calloc
      getV()[trainingLabels[curTrainingLabelIndex]] = strength; // setLabeledNeuron();
      const PVLayerLoc * loc = getLayerLoc();
      status = setActivity_HyPerLayer(getNumNeurons(), getCLayer()->activity->data, getV(), loc->nx, loc->ny, loc->nf, loc->nb);
      // needed because updateState won't call setActivity until the first update period has passed.
      // setActivity();
   }
   return status;
}

int TrainingLayer::updateState(float timef, float dt) {
   int status = PV_SUCCESS;
   if(timef >= nextLabelTime) {
      nextLabelTime += displayPeriod;
      status = updateState(timef, dt, getLayerLoc(), getCLayer()->activity->data, getV(), numTrainingLabels, trainingLabels, curTrainingLabelIndex, strength);
   }
   if( status == PV_SUCCESS ) status = updateActiveIndices();
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
      // return ANNLayer::updateState(time, dt);
   return PV_SUCCESS;
}

int TrainingLayer::checkpointRead(const char * cpDir, float * timef) {
   int status = HyPerLayer::checkpointRead(cpDir, timef);
   assert(status == PV_SUCCESS);
   InterColComm * icComm = parent->icCommunicator();
   int rootProc = 0;
   if (icComm->commRank() == rootProc) {
      char curLabelIndexPath[PV_PATH_MAX];
      int chars_needed = snprintf(curLabelIndexPath, PV_PATH_MAX, "%s/%s_currentLabelIndex.bin", cpDir, name);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "TrainingLayer::checkpointRead error.  Path \"%s/%s_currentLabelIndex.bin\" is too long.\n", cpDir, name);
         abort();
      }
      FILE * curLabelIndexFile = fopen(curLabelIndexPath, "r");
      if (curLabelIndexFile == NULL) {
         fprintf(stderr, "TrainingLayer::checkpointRead error.  Unable to open \"%s\" for reading.  Error %d\n", curLabelIndexPath, errno);
         abort();
      }
      int numread = fread(&curTrainingLabelIndex, sizeof(curTrainingLabelIndex), 1, curLabelIndexFile);
      if (numread != 1) {
         fprintf(stderr, "TrainingLayer::checkpointRead error.  Unable to read \"%s\".\n", curLabelIndexPath);
         abort();
      }
      fclose(curLabelIndexFile);
   }
   MPI_Bcast(&curTrainingLabelIndex, 1, MPI_INT, rootProc, icComm->communicator());
   return status;
}

int TrainingLayer::checkpointWrite(const char * cpDir) {
   int status = HyPerLayer::checkpointWrite(cpDir);
   assert(status == PV_SUCCESS);
   if (parent->icCommunicator()->commRank()==0) {
      char curLabelIndexPath[PV_PATH_MAX];
      int chars_needed = snprintf(curLabelIndexPath, PV_PATH_MAX, "%s/%s_currentLabelIndex.bin", cpDir, name);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "TrainingLayer::checkpointWrite error.  Path \"%s/%s_currentLabelIndex.bin\" is too long.\n", cpDir, name);
         abort();
      }
      FILE * curLabelIndexFile = fopen(curLabelIndexPath, "w");
      if (curLabelIndexFile == NULL) {
         fprintf(stderr, "TrainingLayer::checkpointWrite error.  Unable to open \"%s\" for writing.  Error %d\n", curLabelIndexPath, errno);
         abort();
      }
      int numread = fwrite(&curTrainingLabelIndex, sizeof(curTrainingLabelIndex), 1, curLabelIndexFile);
      if (numread != 1) {
         fprintf(stderr, "TrainingLayer::checkpointWrite error.  Unable to write to \"%s\".\n", curLabelIndexPath);
         abort();
      }
      fclose(curLabelIndexFile);
   }
   return status;
}

} // end of namespace PV
