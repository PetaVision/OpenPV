/*
 * ANNWeightedErrorLayer.cpp
 *
 *  Created on: Jun 21, 2013
 *      Author: gkenyon
 *  uses as input an ascii file storing fraction of ground truth accounted for by each class
 */

#include "ANNWeightedErrorLayer.hpp"
//#include <iostream>
//#include <string>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

  void ANNWeightedErrorLayer_update_state
  (
   const int numNeurons,
   const int nx,
   const int ny,
   const int nf,
   const int lt,
   const int rt,
   const int dn,
   const int up,

   float * V,
   const float Vth,
   const float AMax,
   const float AMin,
   const float AShift,
   float * GSynHead,
   float * activity,
   float errScale, 
   float * errWeights);


#ifdef __cplusplus
}
#endif

namespace PV {

  ANNWeightedErrorLayer::ANNWeightedErrorLayer()
  {
    initialize_base();
  }

  ANNWeightedErrorLayer::ANNWeightedErrorLayer(const char * name, HyPerCol * hc)
  {
    initialize_base();
    initialize(name, hc);
  }

  ANNWeightedErrorLayer::~ANNWeightedErrorLayer()
  {
    free(errWeights);
  }

  int ANNWeightedErrorLayer::initialize_base()
  {
    errScale = 1;
    errWeights = NULL;
    errWeightsFileName = NULL;
    return PV_SUCCESS;
  }

  int ANNWeightedErrorLayer::initialize(const char * name, HyPerCol * hc)
  {
    int status = ANNLayer::initialize(name, hc);
    return status;
  }


  int ANNWeightedErrorLayer::allocateDataStructures() 
  {
    int status = HyPerLayer::allocateDataStructures();
    int nf = getLayerLoc()->nf;
    errWeights = (float *) calloc(nf, sizeof(float *));
    for(int i_weight = 0; i_weight < nf; i_weight++){
      errWeights[i_weight] = 1.0f;
    }
    PV_Stream * pvstream = NULL;
    InterColComm *icComm = getParent()->icCommunicator();
    char errWeight_string[PV_PATH_MAX];
    if (getParent()->icCommunicator()->commRank()==0) {
      PV_Stream * errWeights_stream = pvp_open_read_file(errWeightsFileName, icComm);
      for(int i_weight = 0; i_weight < nf; i_weight++){

	char * fgetsstatus = fgets(errWeight_string, PV_PATH_MAX, errWeights_stream->fp);
	if( fgetsstatus == NULL ) {
	  bool endoffile = feof(errWeights_stream->fp)!=0;
	  if( endoffile ) {
	    fprintf(stderr, "File of errWeights \"%s\" reached end of file before all %d errWeights were read.  Exiting.\n", errWeightsFileName, nf);
	    exit(EXIT_FAILURE);
	  }
	  else {
	    int error = ferror(errWeights_stream->fp);
	    assert(error);
	    fprintf(stderr, "File of errWeights: error %d while reading.  Exiting.\n", error);
	    exit(error);
	  }
	}
	else {
	  // Remove linefeed from end of string
	  errWeight_string[PV_PATH_MAX-1] = '\0';
	  int len = strlen(errWeight_string);
	  if (len > 1) {
	    if (errWeight_string[len-1] == '\n') {
	      errWeight_string[len-1] = '\0';
	    }
	  }
	} // fgetstatus
	
	// set errWeight = chance / relative fraction 
	float errWeight_tmp = atof(errWeight_string);
	fprintf(stderr, "errWeight %i = %f\n", i_weight, errWeight_tmp);
	errWeights[i_weight] = (1.0/nf) / errWeight_tmp;

      } // i_weight
      
    } // commRank() == rootproc
#ifdef PV_USE_MPI
      //broadcast errWeights
    MPI_Bcast(errWeights, nf, MPI_FLOAT, 0, icComm->communicator());
#endif // PV_USE_MPI
     
  } 


  int ANNWeightedErrorLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
    int status = ANNLayer::ioParamsFillGroup(ioFlag);
    ioParam_errScale(ioFlag);
    ioParam_errWeightsFileName(ioFlag);
    return status;
  }

  void ANNWeightedErrorLayer::ioParam_errScale(enum ParamsIOFlag ioFlag) {
    parent->ioParamValue(ioFlag, name, "errScale", &errScale, errScale, true/*warnIfAbsent*/);
  }

  void ANNWeightedErrorLayer::ioParam_errWeightsFileName(enum ParamsIOFlag ioFlag) {
    parent->ioParamString(ioFlag, name, "errWeightsFileName", &errWeightsFileName, NULL, false/*warnIfAbsent*/);
    fprintf(stderr, "%s\n", errWeightsFileName);
    //parent->ioParamStringRequired(ioFlag, errWeightsFileName, "errWeightsFileName", &errWeightsFileName);
  }


  int ANNWeightedErrorLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
					   pvdata_t * V, int num_channels, pvdata_t * gSynHead)
  {
    update_timer->start();
    //#ifdef PV_USE_OPENCL
    //   if(gpuAccelerateFlag) {
    //      updateStateOpenCL(time, dt);
    //      //HyPerLayer::updateState(time, dt);
    //   }
    //   else {
    //#endif
    int nx = loc->nx;
    int ny = loc->ny;
    int nf = loc->nf;
    int num_neurons = nx*ny*nf;
    ANNWeightedErrorLayer_update_state(num_neurons, nx, ny, nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up, V, VThresh,
				       AMax, AMin, AShift, gSynHead, A, errScale, errWeights);
    //#ifdef PV_USE_OPENCL
    //   }
    //#endif

    update_timer->stop();
    return PV_SUCCESS;
  }

} /* namespace PV */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/ANNWeightedErrorLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/ANNWeightedErrorLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif
