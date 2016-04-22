/*
 * InitUniformRandomWeights.hpp
 *
 *  Created on: Aug 9, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMRANDOMWEIGHTS_HPP_
#define INITUNIFORMRANDOMWEIGHTS_HPP_

#include "InitRandomWeights.hpp"
#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

class InitUniformRandomWeights: public PV::InitRandomWeights {
public:
   InitUniformRandomWeights(char const * name, HyPerCol * hc);
   virtual ~InitUniformRandomWeights();

   virtual InitWeightsParams * createNewWeightParams();

protected:
   InitUniformRandomWeights();
   int initialize(char const * name, HyPerCol * hc);
   int randomWeights(pvdata_t * patchDataStart, InitWeightsParams *weightParams, int patchIndex);

private:
   int initialize_base();

}; // class InitUniformRandomWeights

BaseObject * createInitUniformRandomWeights(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
