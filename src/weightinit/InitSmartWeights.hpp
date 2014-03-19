/*
 * InitSmartWeights.hpp
 *
 *  Created on: Aug 8, 2011
 *      Author: kpeterson
 */

#ifndef INITSMARTWEIGHTS_HPP_
#define INITSMARTWEIGHTS_HPP_

#include "InitWeights.hpp"
#include "InitWeightsParams.hpp"

namespace PV {

class InitSmartWeights: public PV::InitWeights {
public:
   InitSmartWeights(HyPerConn * conn);
   InitSmartWeights();
   virtual ~InitSmartWeights();

   virtual InitWeightsParams * createNewWeightParams();

   virtual int calcWeights(/* PVPatch * patch */ pvdata_t * dataStart, int patchIndex, int arborId);

protected:
   int initialize(HyPerConn * conn);

private:
   int initialize_base();
   int smartWeights(/* PVPatch * patch */ pvdata_t * dataStart, int k, InitWeightsParams *weightParams);
};

} /* namespace PV */
#endif /* INITSMARTWEIGHTS_HPP_ */
