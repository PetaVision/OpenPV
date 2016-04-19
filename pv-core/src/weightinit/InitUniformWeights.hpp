/*
 * InitUniformWeights.hpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMWEIGHTS_HPP_
#define INITUNIFORMWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitUniformWeightsParams.hpp"

namespace PV {

class InitUniformWeights: public PV::InitWeights {
public:
   InitUniformWeights(char const * name, HyPerCol * hc);
   virtual ~InitUniformWeights();
   virtual InitWeightsParams * createNewWeightParams();

   virtual int calcWeights(pvdata_t * dataStart, int patchIndex, int arborId);

protected:
   InitUniformWeights();
   int initialize_base();
   int initialize(char const * name, HyPerCol * hc);

private:
  int uniformWeights(pvdata_t * dataStart, float iWeight, int kf, InitUniformWeightsParams *weightParamPtr, bool connectOnlySameFeatures = false);

}; // class InitUniformWeights

BaseObject * createInitUniformWeights(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* INITUNIFORMWEIGHTS_HPP_ */
