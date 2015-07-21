/*
 * OjaSTDPConn.hpp
 *
 *  Created on: Sep 27, 2012
 *      Author: dpaiton et slundquist
 */

#ifndef OJASTDPCONN_HPP_
#define OJASTDPCONN_HPP_

//#define SPLIT_PRE_POST
#undef SPLIT_PRE_POST

#include "HyPerConn.hpp"
#include "../include/default_params.h"
#include <stdio.h>
#include "../io/fileio.hpp"

namespace PV {

class OjaSTDPConn: public HyPerConn {
public:
   OjaSTDPConn();
   OjaSTDPConn(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual ~OjaSTDPConn();

   virtual int communicateInitInfo();
   virtual int allocateDataStructures();

   virtual float maxWeight(int axonID);
   virtual int writeTextWeightsExtra(PV_Stream * pvstream, int k, int axonID);

   virtual int updateState(double time, double dt);
   virtual int updateAmpLTD();
   virtual int updateWeights(int axonID);
   virtual int outputState(double time, bool last=false);

   virtual int checkpointRead(const char * cpDir, double * timef);
   virtual int checkpointWrite(const char * cpDir);

   //get functions
   float getPostStdpTr(int k)  {return post_stdp_tr->data[k];}
   float getPostOjaTr(int k)   {return post_oja_tr->data[k];}
   float getPostIntTr(int k)   {return post_int_tr->data[k];}
   float getAmpLTD(int k)      {return ampLTD[k];}
   float getPreStdpTr(int kex,int arborID) {return pre_stdp_tr[arborID]->data[kex];}
   float getPreOjaTr(int kex,int arborID)  {return pre_oja_tr[arborID]->data[kex];}

   pvwdata_t ** getPostWeightsp(int axonID, int kPost);

   int getNxpPost() {return nxpPost;}
   int getNypPost() {return nypPost;}
   int getNfpPost() {return nfpPost;}

protected:

   int initialize_base();
   int initialize(const char * name, HyPerCol * hc, InitWeights * weightInitializer=NULL, NormalizeBase * weightNormalizer=NULL);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_ampLTP(enum ParamsIOFlag ioFlag);
   virtual void ioParam_initAmpLTD(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauLTP(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauLTD(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauOja(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauTHR(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauO(enum ParamsIOFlag ioFlag);
   virtual void ioParam_targetPostRate(enum ParamsIOFlag ioFlag);
   virtual void ioParam_ojaFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_synscalingFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_synscaling_v(enum ParamsIOFlag ioFlag);
   virtual void ioParam_wMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_wMin(enum ParamsIOFlag ioFlag);
   virtual void ioParam_weightScale(enum ParamsIOFlag ioFlag);
   virtual void ioParam_LTDscale(enum ParamsIOFlag ioFlag);

   virtual int initPlasticityPatches();
#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers(const char * kernelName);
   virtual int initializeThreadKernels(const char * kernelName);
#endif // PV_USE_OPENCL

   virtual int scaleWeights();

   PVLayerCube * post_stdp_tr; // plasticity decrement variable for postsynaptic layer
   PVLayerCube * post_oja_tr;  // plasticity decrement variable for longer time-constant
   PVLayerCube * post_int_tr;  // plasticity decrement variable for longer time-constant

   //Need pre trace per arbor
   PVLayerCube ** pre_stdp_tr;  // plasticity increment variable for presynaptic layer
   PVLayerCube ** pre_oja_tr;   // plasticity increment variable for presynaptic layer with longer time-constant
   MPI_Datatype * mpi_datatype;  // Used to mirror the presynaptic traces

   float * ampLTD;

   // STDP parameters for modifying weights
   float ampLTP; // long term potentiation amplitude
   float initAmpLTD; // long term depression amplitude
   float tauLTP;
   float tauLTD;
   float tauOja;
   float tauTHR;
   float tauO;
   float targetPostRateHz;
   float weightScale;
   float LTDscale;

   bool  ojaFlag;
   bool  synscalingFlag;
   float synscaling_v;

private:
   int deleteWeights();

};

}

#endif /* OJASTDPCONN_HPP_ */
