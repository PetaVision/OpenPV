/*
 * InitMaxPoolingWeights.hpp
 *
 *  Created on: Jan 15, 2015
 *      Author: gkenyon
 */

#ifndef INITMAXPOOLINGWEIGHTS_HPP_
#define INITMAXPOOLINGWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitMaxPoolingWeightsParams.hpp"

namespace PV {

class InitMaxPoolingWeights: public PV::InitWeights {
public:
   InitMaxPoolingWeights(HyPerConn * conn);
   virtual ~InitMaxPoolingWeights();
   virtual InitWeightsParams * createNewWeightParams();

   virtual int calcWeights(pvdata_t * dataStart, int patchIndex, int arborId);

protected:
   InitMaxPoolingWeights();
   int initialize_base();
   int initialize(HyPerConn * conn);


};

} /* namespace PV */
#endif /* INITMAXPOOLINGWEIGHTS_HPP_ */
