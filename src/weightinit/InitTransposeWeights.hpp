/*
 * InitTransposeWeights.hpp
 *
 *  Created on: Aug 15, 2011
 *      Author: kpeterson
 */

#ifndef INITTRANSPOSEWEIGHTS_HPP_
#define INITTRANSPOSEWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "../connections/KernelConn.hpp"

namespace PV {

class KernelConn;
class InitWeightsParams;
class InitTransposeWeightsParams;

class InitTransposeWeights: public PV::InitWeights {
public:
   InitTransposeWeights();
   InitTransposeWeights(KernelConn * origConn);
   virtual ~InitTransposeWeights();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex,
         InitWeightsParams *weightParams);
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);
   // void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   virtual int initialize_base();
   int initialize(KernelConn * origConn);

private:
   int transposeKernels(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, InitTransposeWeightsParams * weightParamPtr);

   KernelConn * originalConn;

};

} /* namespace PV */
#endif /* INITTRANSPOSEWEIGHTS_HPP_ */
