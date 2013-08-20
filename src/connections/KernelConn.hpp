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

   KernelConn(const char * name, HyPerCol * hc, const char * pre_layer_name,
         const char * post_layer_name, const char * filename = NULL,
         InitWeights *weightInit = NULL);
   virtual int allocateDataStructures();

   virtual float minWeight(int arborId = 0);
   virtual float maxWeight(int arborId = 0);

   //virtual int checkNormalizeArbor(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);
   // virtual int normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);
   // virtual int symmetrizeWeights(pvdata_t * dataStart, int numPatches, int arborId);

   virtual int writeWeights(double time, bool last=false);
   virtual int writeWeights(const char * filename);

   bool getPlasticityFlag() {return plasticityFlag;}
   float getWeightUpdatePeriod() {return weightUpdatePeriod;}
   float getWeightUpdateTime() {return weightUpdateTime;}
   float getLastUpdateTime() {return lastUpdateTime;}


   virtual int checkpointWrite(const char * cpDir);
   virtual int checkpointRead(const char * cpDir, double *timef);

#ifdef PV_USE_OPENCL
   virtual int * getLUTpointer() {return patch2datalookuptable;}
#endif // PV_USE_OPENCL
   virtual void initPatchToDataLUT();
   virtual int patchToDataLUT(int patchIndex);
   virtual int patchIndexToDataIndex(int patchIndex, int * kx=NULL, int * ny=NULL, int * nf=NULL);
   virtual int dataIndexToUnitCellIndex(int dataIndex, int * kx=NULL, int * ny=NULL, int * nf=NULL);


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
                  const char * pre_layer_name, const char * post_layer_name,
                  const char * filename, InitWeights *weightInit=NULL);
   virtual int communicateInitInfo();
   virtual int createArbors();
   virtual int initPlasticityPatches();
   //virtual pvdata_t * allocWeights(PVPatch *** patches, int nPatches, int nxPatch,
   //      int nyPatch, int nfPatch, int arborId);
   virtual pvdata_t * allocWeights(int nPatches, int nxPatch, int nyPatch, int nfPatch);
   int initNumDataPatches();
#ifdef OBSOLETE // Marked obsolete April 15, 2013.  Implementing the new NormalizeBase class hierarchy
   virtual int initNormalize();
#endif // OBSOLETE
   virtual PVPatch *** initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart,
         int numPatches, const char * filename);

   virtual int calc_dW(int arborId);
   virtual int clear_dW(int arborId);
   virtual int update_dW(int arborId);
   virtual int defaultUpdate_dW(int arborId);
   virtual pvdata_t updateRule_dW(pvdata_t pre, pvdata_t post);
   virtual bool skipPre(pvdata_t preact){return preact == 0.0f;};

   virtual int updateState(double time, double dt);
   virtual int updateWeights(int arborId);
   virtual float computeNewWeightUpdateTime(double time, double currentUpdateTime);
#ifdef PV_USE_MPI
   virtual int reduceKernels(int arborID);
#else
   virtual int reduceKernels(int arborID){return PV_SUCCESS;};
#endif // PV_USE_MPI
//   virtual PVPatch ** readWeights(PVPatch ** patches, int numPatches,
//                                     const char * filename);
   int getReciprocalWgtCoordinates(int kx, int ky, int kf, int kernelidx, int * kxRecip, int * kyRecip, int * kfRecip, int * kernelidxRecip);

   virtual int setParams(PVParams* params);
   virtual void readShmget_flag(PVParams * params);
   virtual void readKeepKernelsSynchronized(PVParams * params);
   virtual void readWeightUpdatePeriod(PVParams * params);
   virtual void readInitialWeightUpdateTime(PVParams * params);
   virtual void readUseWindowPost(PVParams * params);

private:
   int deleteWeights();
};

}

#endif /* KERNELCONN_HPP_ */
