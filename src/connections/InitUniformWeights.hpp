/*
 * InitUniformWeights.hpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMWEIGHTS_HPP_
#define INITUNIFORMWEIGHTS_HPP_

#include "InitWeights.hpp"

namespace PV {

class InitUniformWeights: public PV::InitWeights {
public:
   InitUniformWeights();
   virtual ~InitUniformWeights();
   virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

   virtual int calcWeights(PVPatch * patch, int patchIndex, int arborId, InitWeightsParams *weightParams);

protected:
   virtual int initialize_base();

private:
   int uniformWeights(PVPatch * wp, float iWeight);

};

} /* namespace PV */
#endif /* INITUNIFORMWEIGHTS_HPP_ */
