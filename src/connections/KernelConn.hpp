/*
 * KernelConn.hpp
 *
 *  Created on: Aug 6, 2009
 *      Author: gkenyon
 */

#ifndef KERNELCONN_HPP_
#define KERNELCONN_HPP_

#include "HyPerConn.hpp"
#ifdef PV_USE_MPI
#  include <mpi.h>
#else
#  include "../include/mpi_stubs.h"
#endif // PV_USE_MPI

namespace PV {

class KernelConn: public HyPerConn {

public:

   KernelConn();
   virtual ~KernelConn();

   KernelConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
              const char * filename = NULL, InitWeights *weightInit = NULL);

   virtual float minWeight(int axonId = 0);
   virtual float maxWeight(int axonId = 0);

   //virtual int checkNormalizeArbor(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);
   virtual int normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);
   virtual int symmetrizeWeights(pvdata_t * dataStart, int numPatches, int arborId);

   virtual int writeWeights(float time, bool last=false);
   virtual int writeWeights(const char * filename);

   bool getPlasticityFlag() {return plasticityFlag;}
   float getWeightUpdatePeriod() {return weightUpdatePeriod;}
   float getWeightUpdateTime() {return weightUpdateTime;}
   float getLastUpdateTime() {return lastUpdateTime;}


   virtual int checkpointWrite(const char * cpDir);
   virtual int checkpointRead(const char * cpDir, float *timef);

#ifdef PV_USE_OPENCL
   virtual int * getLUTpointer() {return patch2datalookuptable;}
#endif // PV_USE_OPENCL
   void initPatchToDataLUT();
   virtual int patchToDataLUT(int patchIndex);
   virtual int patchIndexToDataIndex(int patchIndex, int * kx=NULL, int * ny=NULL, int * nf=NULL);
   virtual int dataIndexToUnitCellIndex(int dataIndex, int * kx=NULL, int * ny=NULL, int * nf=NULL);

#ifdef USE_SHMGET
    virtual bool getShmgetOwner(){
      return shmget_owner;
   };
#endif


protected:
   float weightUpdatePeriod;
   float weightUpdateTime;
   float lastUpdateTime;
   bool symmetrizeWeightsFlag;


private:
   int * patch2datalookuptable;


protected:
//   int nxKernel;
//   int nyKernel;
//   int nfKernel;

#ifdef PV_USE_MPI
   pvdata_t * mpiReductionBuffer;
   bool keepKernelsSynchronized_flag;
#endif // PV_USE_MPI

   int initialize_base();
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  const char * filename,
                  InitWeights *weightInit=NULL);
   virtual int createArbors();
   virtual int initPlasticityPatches();
   virtual pvdata_t * allocWeights(PVPatch *** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   int initNumDataPatches();
   virtual int initializeUpdateTime(PVParams * params);
   virtual PVPatch *** initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart,
         int numPatches, const char * filename);

   virtual int calc_dW(int axonId);
   virtual int clear_dW(int axonId);
   virtual int update_dW(int axonId);
   virtual int defaultUpdate_dW(int axonId);
   virtual pvdata_t updateRule_dW(pvdata_t pre, pvdata_t post);

   virtual int updateState(float time, float dt);
   virtual int updateWeights(int axonId);
   virtual float computeNewWeightUpdateTime(float time, float currentUpdateTime);
#ifdef PV_USE_MPI
   virtual int reduceKernels(int axonID);
#endif // PV_USE_MPI
//   virtual PVPatch ** readWeights(PVPatch ** patches, int numPatches,
//                                     const char * filename);
   int getReciprocalWgtCoordinates(int kx, int ky, int kf, int kernelidx, int * kxRecip, int * kyRecip, int * kfRecip, int * kernelidxRecip);

private:
   int deleteWeights();
};

}

#endif /* KERNELCONN_HPP_ */
