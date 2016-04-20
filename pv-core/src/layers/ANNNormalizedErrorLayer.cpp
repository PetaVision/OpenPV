/*
 * ANNNormalizedErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ANNNormalizedErrorLayer.hpp"
#include <fstream>

namespace PV {

ANNNormalizedErrorLayer::ANNNormalizedErrorLayer()
{
   initialize_base();
}

ANNNormalizedErrorLayer::ANNNormalizedErrorLayer(const char * name, HyPerCol * hc)
{
   initialize_base();
   initialize(name, hc);
}

ANNNormalizedErrorLayer::~ANNNormalizedErrorLayer()
{
   if (parent->getDtAdaptFlag() && parent->getWriteTimescales()){
      timeScaleStream.close();
   }
   if(maskLayerName){
      free(maskLayerName);
   }
   if(timeScale){
      free(timeScale);
   }
}

int ANNNormalizedErrorLayer::initialize_base()
{
   timeScale = NULL;
   useMask = false;
   maskLayerName = NULL;
   maskLayer = NULL;
   return PV_SUCCESS;
}

int ANNNormalizedErrorLayer::initialize(const char * name, HyPerCol * hc)
{
   if (hc->columnId()==0) {
      // ANNNormalizedErrorLayer was deprecated on Feb 1, 2016.
      fflush(stdout);
      fprintf(stderr, "\n\nWarning: ANNNormalizedErrorLayer is deprecated.\n");
      fprintf(stderr, "If you are using this class to control an adaptive timestep, define a dtAdaptControlProbe instead.\n\n\n");
   }
   int status = ANNErrorLayer::initialize(name, hc);

   if (parent->icCommunicator()->commRank()==0 && parent->getDtAdaptFlag() && parent->getWriteTimescales()){
      char * outputPath = parent->getOutputPath();
      size_t timeScaleFileNameLen = strlen(outputPath) + strlen("/") + strlen(name) + strlen("_timescales.txt");
      char timeScaleFileName[timeScaleFileNameLen+1];
      int charsneeded = snprintf(timeScaleFileName, timeScaleFileNameLen+1, "%s/%s_timescales.txt", outputPath, name);
      assert(charsneeded<=timeScaleFileNameLen);
      timeScaleStream.open(timeScaleFileName);
      timeScaleStream.precision(17);
   }
   return status;
}

double ANNNormalizedErrorLayer::getTimeScale(int batchIdx){
   assert(batchIdx >= 0 && batchIdx < parent->getNBatch());
   return timeScale[batchIdx];
}

double ANNNormalizedErrorLayer::calcTimeScale(int batchIdx){
   assert(batchIdx >= 0 && batchIdx < parent->getNBatch());
   if (parent->getDtAdaptFlag()){
      timescale_timer->start();
      InterColComm * icComm = parent->icCommunicator();
      //int num_procs = icComm->numCommColumns() * icComm->numCommRows();
      int num_neurons = getNumNeurons();
      pvdata_t * gSynExc = getChannel(CHANNEL_EXC);
      pvdata_t * gSynInh = getChannel(CHANNEL_INH);
      pvdata_t * gSynExcBatch = gSynExc + batchIdx * num_neurons;
      pvdata_t * gSynInhBatch = gSynInh + batchIdx * num_neurons;

      pvdata_t * maskActivityBatch = NULL;
      if(useMask){
         maskActivityBatch = maskLayer->getActivity() + batchIdx * num_neurons;
      }

      double errorMag = 0;
      double inputMag = 0;

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : errorMag, inputMag)
#endif
      for (int ni = 0; ni < num_neurons; ni++){
         if(useMask){
            int kMaskRes;
            const PVLayerLoc * maskLoc = maskLayer->getLayerLoc();
            if(maskLoc->nf == 1){
               kMaskRes = ni/getLayerLoc()->nf;
            }
            else{
               kMaskRes = ni;
            }
            int kMaskExt = kIndexExtended(ni, maskLoc->nx, maskLoc->ny, maskLoc->nf, maskLoc->halo.lt, maskLoc->halo.rt, maskLoc->halo.dn, maskLoc->halo.up);
            //If value is masked out, do not add to errorMag/inputMag
            if(maskActivityBatch[kMaskExt] == 0){
               continue; 
            }
         }
         errorMag += (gSynExcBatch[ni] - gSynInhBatch[ni]) * (gSynExcBatch[ni] - gSynInhBatch[ni]);
         inputMag += gSynExcBatch[ni] * gSynExcBatch[ni]; 
      }
      if (isnan(errorMag)) {
         fprintf(stderr, "Layer \"%s\": errorMag on process %d is not a number.\n", getName(), getParent()->columnId());
         exit(EXIT_FAILURE);
      }
      else if (errorMag < 0) {
         fprintf(stderr, "Layer \"%s\": errorMag on process %d is negative.  This should absolutely never happen.\n", getName(), getParent()->columnId());
         exit(EXIT_FAILURE);
      }
      //Sum all errMag across processors
      MPI_Allreduce(MPI_IN_PLACE, &errorMag, 1, MPI_DOUBLE, MPI_SUM, icComm->communicator());
      MPI_Allreduce(MPI_IN_PLACE, &inputMag, 1, MPI_DOUBLE, MPI_SUM, icComm->communicator());
      //errorMag /= num_neurons * num_procs;
      //inputMag /= num_neurons * num_procs;
      timeScale[batchIdx] = errorMag > 0 ? sqrt(inputMag / errorMag) : parent->getTimeScaleMin();
      //fprintf(stdout, "timeScale: %f\n", timeScale);
      timescale_timer->stop();
      if (parent->getWriteTimescales() && parent->icCommunicator()->commRank()==0){
         timeScaleStream << "sim_time = " << parent->simulationTime() << ", " << "timeScale = " << timeScale[batchIdx] << std::endl; }
      return timeScale[batchIdx];
   }
   else{
      return HyPerLayer::calcTimeScale(batchIdx);
   }
}

//TODO take this out, this is a hack to get masking working with NormalizedErrorLayer
int ANNNormalizedErrorLayer::updateState(double time, double dt){
   ANNErrorLayer::updateState(time, dt);
   if(!useMask){return PV_SUCCESS;}

   const PVLayerLoc * loc = getLayerLoc();
   const PVLayerLoc * maskLoc = maskLayer->getLayerLoc();
   pvdata_t * maskActivity = maskLayer->getActivity();
   pvdata_t * A = getActivity();

   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int num_neurons = nx*ny*nf;
   int nbatch = loc->nbatch;

   for(int b = 0; b < nbatch; b++){
      pvdata_t * maskActivityBatch = maskActivity + b * num_neurons;
      pvdata_t * ABatch = A + b * getNumExtended();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
      for(int ni = 0; ni < num_neurons; ni++){
         int kThisRes = ni;
         int kThisExt = kIndexExtended(ni, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
         int kMaskRes;
         if(maskLoc->nf == 1){
            kMaskRes = ni/nf;
         }
         else{
            kMaskRes = ni;
         }
         int kMaskExt = kIndexExtended(ni, nx, ny, maskLoc->nf, maskLoc->halo.lt, maskLoc->halo.rt, maskLoc->halo.dn, maskLoc->halo.up);
         //Set value to 0, otherwise, updateState from ANNLayer should have taken care of it
         if(maskActivityBatch[kMaskExt] == 0){
            ABatch[kThisExt] = 0;
         }
      }
   }
   return PV_SUCCESS;
}

int ANNNormalizedErrorLayer::allocateDataStructures() {
   int status = ANNErrorLayer::allocateDataStructures();
   timeScale = (double*) malloc(sizeof(double) * parent->getNBatch());
   assert(timeScale);
   //Initialize timeScales to 1
   for(int b = 0; b < parent->getNBatch(); b++){
      timeScale[b] = 1;
   }
   return status;
}

//TODO take this out, this is a hack to get masking working with NormalizedErrorLayer
int ANNNormalizedErrorLayer::communicateInitInfo() {
   int status = ANNErrorLayer::communicateInitInfo();
   if(useMask){
      maskLayer = parent->getLayerFromName(maskLayerName);
      if (maskLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" is not a layer in the HyPerCol.\n",
                    getKeyword(), name, maskLayerName);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      const PVLayerLoc * maskLoc = maskLayer->getLayerLoc();
      const PVLayerLoc * loc = getLayerLoc();
      assert(maskLoc != NULL && loc != NULL);
      if (maskLoc->nxGlobal != loc->nxGlobal || maskLoc->nyGlobal != loc->nyGlobal) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" does not have the same x and y dimensions.\n",
                    getKeyword(), name, maskLayerName);
            fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                    maskLoc->nxGlobal, maskLoc->nyGlobal, maskLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      if(maskLoc->nf != 1 && maskLoc->nf != loc->nf){
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" must either have the same number of features as this layer, or one feature.\n",
                    getKeyword(), name, maskLayerName);
            fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                    maskLoc->nxGlobal, maskLoc->nyGlobal, maskLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         }
         MPI_Barrier(parent->icCommunicator()->communicator());
         exit(EXIT_FAILURE);
      }

      assert(maskLoc->nx==loc->nx && maskLoc->ny==loc->ny);
   }

   return status;
}

int ANNNormalizedErrorLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNErrorLayer::ioParamsFillGroup(ioFlag);
   ioParam_useMask(ioFlag);
   ioParam_maskLayerName(ioFlag);
   return status;
}

void ANNNormalizedErrorLayer::ioParam_useMask(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "useMask", &useMask, useMask, false/*warnIfAbsent*/);
}

void ANNNormalizedErrorLayer::ioParam_maskLayerName(enum ParamsIOFlag ioFlag) {
   if(useMask){
      parent->ioParamStringRequired(ioFlag, name, "maskLayerName", &maskLayerName);
   }
}

BaseObject * createANNNormalizedErrorLayer(char const * name, HyPerCol * hc) {
   return hc ? new ANNNormalizedErrorLayer(name, hc) : NULL;
}

} /* namespace PV */

