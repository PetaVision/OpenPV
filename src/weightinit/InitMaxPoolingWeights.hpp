/*
 * InitMaxPoolingWeights.hpp
 *
 *  Created on: Jan 15, 2015
 *      Author: gkenyon
 */

#ifndef INITMAXPOOLINGWEIGHTS_HPP_
#define INITMAXPOOLINGWEIGHTS_HPP_

#include "InitMaxPoolingWeightsParams.hpp"
#include "InitWeights.hpp"

namespace PV {

class InitMaxPoolingWeights : public PV::InitWeights {
  public:
   InitMaxPoolingWeights(char const *name, HyPerCol *hc);
   virtual ~InitMaxPoolingWeights();
   virtual InitWeightsParams *createNewWeightParams();

   virtual int calcWeights(float *dataStart, int patchIndex, int arborId);

  protected:
   InitMaxPoolingWeights();
   int initialize_base();
   int initialize(char const *name, HyPerCol *hc);

}; // class InitMaxPoolingWeights

} /* namespace PV */
#endif /* INITMAXPOOLINGWEIGHTS_HPP_ */
