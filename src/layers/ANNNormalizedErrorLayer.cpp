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
}

int ANNNormalizedErrorLayer::initialize_base()
{
   timeScale = 1;
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
#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for reduction(+ : errorMag, inputMag)
#endif

      for (int i = 0; i < num_neurons; i++){
         errorMag += (gSynExc[i] - gSynInh[i]) * (gSynExc[i] - gSynInh[i]);
         inputMag += gSynExc[i] * gSynExc[i]; 
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


} /* namespace PV */

