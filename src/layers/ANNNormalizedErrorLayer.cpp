/*
 * ANNNormalizedErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 */

#include "ANNNormalizedErrorLayer.hpp"

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
  }

  int ANNNormalizedErrorLayer::initialize_base()
  {
    timeScale = 1;
    return PV_SUCCESS;
  }

  int ANNNormalizedErrorLayer::initialize(const char * name, HyPerCol * hc)
  {
    int status = ANNErrorLayer::initialize(name, hc);
    return status;
  }

  double ANNNormalizedErrorLayer::getTimeScale(){
    InterColComm * icComm = parent->icCommunicator();
    //int num_procs = icComm->numCommColumns() * icComm->numCommRows();
    int num_neurons = getNumNeurons();
    double errorMag = 0;
    double inputMag = 0;
    pvdata_t * gSynExc = getChannel(CHANNEL_EXC);
    pvdata_t * gSynInh = getChannel(CHANNEL_INH);
    for (int i = 0; i < num_neurons; i++){
      errorMag += (gSynExc[i] - gSynInh[i]) * (gSynExc[i] - gSynInh[i]);
      inputMag += gSynExc[i] * gSynExc[i];
    }
#ifdef PV_USE_MPI
    //Sum all errMag across processors
    MPI_Allreduce(MPI_IN_PLACE, &errorMag, 1, MPI_DOUBLE, MPI_SUM, icComm->communicator());
    MPI_Allreduce(MPI_IN_PLACE, &inputMag, 1, MPI_DOUBLE, MPI_SUM, icComm->communicator());
#endif // PV_USE_MPI
    //errorMag /= num_neurons * num_procs;
    //inputMag /= num_neurons * num_procs;
    timeScale = errorMag > 0 ? sqrt(inputMag / errorMag) : 1.0;
    return timeScale;
  }


} /* namespace PV */

