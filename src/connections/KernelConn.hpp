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

   virtual ~KernelConn();

   KernelConn(const char * name, HyPerCol * hc);
   virtual int allocateDataStructures();

   virtual float minWeight(int arborId = 0);
   virtual float maxWeight(int arborId = 0);

   //virtual int checkNormalizeArbor(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);
   // virtual int normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);
   // virtual int symmetrizeWeights(pvdata_t * dataStart, int numPatches, int arborId);

   virtual int writeWeights(double time, bool last=false);
   virtual int writeWeights(const char * filename);

   bool getPlasticityFlag() {return plasticityFlag;}

   //Moved to HyPerConn
   //double getWeightUpdatePeriod() {return weightUpdatePeriod;}
   //double getWeightUpdateTime() {return weightUpdateTime;}
   //double getLastUpdateTime() {return lastUpdateTime;}


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
   //Moved to HyPerConn
   //double weightUpdatePeriod;
   //double weightUpdateTime;
   //double initialWeightUpdateTime;
   //double lastUpdateTime;
   bool symmetrizeWeightsFlag;
   int* numKernelActivations;


private:
   int * patch2datalookuptable;


protected:
//   int nxKernel;
//   int nyKernel;
//   int nfKernel;

#if PV_USE_MPI
   pvdata_t * mpiReductionBuffer;
#endif
   bool keepKernelsSynchronized_flag;

   KernelConn();
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc);
   virtual int communicateInitInfo();
   virtual int createArbors();
   virtual int initPlasticityPatches();
   //virtual pvdata_t * allocWeights(PVPatch *** patches, int nPatches, int nxPatch,
   //      int nyPatch, int nfPatch, int arborId);
   virtual pvdata_t * allocWeights(int nPatches, int nxPatch, int nyPatch, int nfPatch);
   int initNumDataPatches();
   virtual PVPatch *** initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart,
         int numPatches);

   virtual int calc_dW(int arborId);
   virtual int clear_dW(int arborId);
   virtual int update_dW(int arborId);
   virtual int defaultUpdate_dW(int arborId);
   virtual int defaultUpdateInd_dW(int arbor_ID, int kExt);
   virtual int normalize_dW(int arbor_ID);
   virtual pvdata_t updateRule_dW(pvdata_t pre, pvdata_t post);
   virtual bool skipPre(pvdata_t preact){return preact == 0.0f;};

   virtual int updateState(double time, double dt);
   virtual int updateWeights(int arborId);
   //Moved to HyPerConn
   //virtual double computeNewWeightUpdateTime(double time, double currentUpdateTime);
#ifdef PV_USE_MPI
   virtual int reduceKernels(int arborID);
#else
   virtual int reduceKernels(int arborID){return PV_SUCCESS;};
#endif // PV_USE_MPI
//   virtual PVPatch ** readWeights(PVPatch ** patches, int numPatches,
//                                     const char * filename);
   int getReciprocalWgtCoordinates(int kx, int ky, int kf, int kernelidx, int * kxRecip, int * kyRecip, int * kfRecip, int * kernelidxRecip);

   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_shmget_flag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_keepKernelsSynchronized(enum ParamsIOFlag ioFlag);
   //Moved to HyPerConn
   //virtual void ioParam_weightUpdatePeriod(enum ParamsIOFlag ioFlag);
   //virtual void ioParam_initialWeightUpdateTime(enum ParamsIOFlag ioFlag);
   virtual void ioParam_useWindowPost(enum ParamsIOFlag ioFlag);

private:
   int deleteWeights();
};

}

#endif /* KERNELCONN_HPP_ */
