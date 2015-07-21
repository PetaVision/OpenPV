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
}

int ANNNormalizedErrorLayer::initialize_base()
{
   timeScale = 1;
   useMask = false;
   maskLayerName = NULL;
   maskLayer = NULL;
   return PV_SUCCESS;
}

int ANNNormalizedErrorLayer::initialize(const char * name, HyPerCol * hc)
{
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

double ANNNormalizedErrorLayer::getTimeScale(){
   return timeScale;
}

double ANNNormalizedErrorLayer::calcTimeScale(){
   if (parent->getDtAdaptFlag()){
      timescale_timer->start();
      InterColComm * icComm = parent->icCommunicator();
      //int num_procs = icComm->numCommColumns() * icComm->numCommRows();
      int num_neurons = getNumNeurons();
      double errorMag = 0;
      double inputMag = 0;
      pvdata_t * gSynExc = getChannel(CHANNEL_EXC);
      pvdata_t * gSynInh = getChannel(CHANNEL_INH);

      pvdata_t * maskActivity = NULL;
      if(useMask){
         maskActivity = maskLayer->getActivity();
      }

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
            if(maskActivity[kMaskExt] == 0){
               continue; 
            }
         }
         errorMag += (gSynExc[ni] - gSynInh[ni]) * (gSynExc[ni] - gSynInh[ni]);
         inputMag += gSynExc[ni] * gSynExc[ni]; 
      }
      if (isnan(errorMag)) {
         fprintf(stderr, "Layer \"%s\": errorMag on process %d is not a number.\n", getName(), getParent()->columnId());
         exit(EXIT_FAILURE);
      }
      else if (errorMag < 0) {
         fprintf(stderr, "Layer \"%s\": errorMag on process %d is negative.  This should absolutely never happen.\n", getName(), getParent()->columnId());
         exit(EXIT_FAILURE);
      }
#ifdef PV_USE_MPI
      //Sum all errMag across processors
      MPI_Allreduce(MPI_IN_PLACE, &errorMag, 1, MPI_DOUBLE, MPI_SUM, icComm->communicator());
      MPI_Allreduce(MPI_IN_PLACE, &inputMag, 1, MPI_DOUBLE, MPI_SUM, icComm->communicator());
#endif // PV_USE_MPI
      //errorMag /= num_neurons * num_procs;
      //inputMag /= num_neurons * num_procs;
      timeScale = errorMag > 0 ? sqrt(inputMag / errorMag) : parent->getTimeScaleMin();
      //fprintf(stdout, "timeScale: %f\n", timeScale);
      timescale_timer->stop();
      if (parent->getWriteTimescales() && parent->icCommunicator()->commRank()==0){
         timeScaleStream << "sim_time = " << parent->simulationTime() << ", " << "timeScale = " << timeScale << std::endl;
      }
      return timeScale;
   }
   else{
      return HyPerLayer::calcTimeScale();
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
      if(maskActivity[kMaskExt] == 0){
         A[kThisExt] = 0;
      }
   }
   return PV_SUCCESS;
}

//TODO take this out, this is a hack to get masking working with NormalizedErrorLayer
int ANNNormalizedErrorLayer::communicateInitInfo() {
   int status = ANNErrorLayer::communicateInitInfo();
   if(useMask){
      maskLayer = parent->getLayerFromName(maskLayerName);
      if (maskLayer==NULL) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" is not a layer in the HyPerCol.\n",
                    parent->parameters()->groupKeywordFromName(name), name, maskLayerName);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }

      const PVLayerLoc * maskLoc = maskLayer->getLayerLoc();
      const PVLayerLoc * loc = getLayerLoc();
      assert(maskLoc != NULL && loc != NULL);
      if (maskLoc->nxGlobal != loc->nxGlobal || maskLoc->nyGlobal != loc->nyGlobal) {
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" does not have the same x and y dimensions.\n",
                    parent->parameters()->groupKeywordFromName(name), name, maskLayerName);
            fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                    maskLoc->nxGlobal, maskLoc->nyGlobal, maskLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
         exit(EXIT_FAILURE);
      }

      if(maskLoc->nf != 1 && maskLoc->nf != loc->nf){
         if (parent->columnId()==0) {
            fprintf(stderr, "%s \"%s\" error: maskLayerName \"%s\" must either have the same number of features as this layer, or one feature.\n",
                    parent->parameters()->groupKeywordFromName(name), name, maskLayerName);
            fprintf(stderr, "    original (nx=%d, ny=%d, nf=%d) versus (nx=%d, ny=%d, nf=%d)\n",
                    maskLoc->nxGlobal, maskLoc->nyGlobal, maskLoc->nf, loc->nxGlobal, loc->nyGlobal, loc->nf);
         }
#ifdef PV_USE_MPI
         MPI_Barrier(parent->icCommunicator()->communicator());
#endif
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


} /* namespace PV */

