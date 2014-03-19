/*
 * InitBIDSLateral.hpp
 *
 *  Created on: Aug 10, 2012
 *      Author: bnowers
 */

#ifndef INITBIDSLATERAL_HPP_
#define INITBIDSLATERAL_HPP_

#include "../include/pv_common.h"
#include "../include/pv_types.h"
#include "../io/PVParams.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../layers/BIDSMovieCloneMap.hpp"
#include "InitWeightsParams.hpp"
#include "InitBIDSLateralParams.hpp"

namespace PV {

class HyPerCol;
class HyPerLayer;
class InitWeightsParams;
class InitGauss2DWeightsParams;

class InitBIDSLateral: public PV::InitWeights {
public:
   InitBIDSLateral(HyPerConn * conn);
   virtual ~InitBIDSLateral();

   virtual InitWeightsParams * createNewWeightParams();

   virtual int calcWeights(pvdata_t * dataStart, int patchIndex, int arborId);


   //get-set methods:
//   inline const char * getName()                     {return name;}


protected:
   InitBIDSLateral();
   int initialize(HyPerConn * conn);

private:
   int initialize_base();
   int BIDSLateralCalcWeights(/* PVPatch * patch */ int kPre, pvdata_t * dataStart, InitBIDSLateralParams * weightParamPtr);
};

} /* namespace PV */
#endif /* INITBIDSLATERAL_HPP_ */
