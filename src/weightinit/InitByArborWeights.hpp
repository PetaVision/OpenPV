/*
 * InitByArbor.hpp
 *
 *      Author: slundquist 
 */

#ifndef INITBYARBORWEIGHTS_HPP_
#define INITBYARBORWEIGHTS_HPP_

#include "InitWeights.hpp"
//#include "InitUniformRandomWeightsParams.hpp"

namespace PV {

class InitByArborWeights: public PV::InitWeights {
public:
   InitByArborWeights();
   virtual ~InitByArborWeights();

   //virtual InitWeightsParams * createNewWeightParams(HyPerConn * callingConn);

   virtual int calcWeights(/* PVPatch * wp */ pvdata_t * dataStart, int patchIndex, int arborId, InitWeightsParams *weightParams);

protected:
   int initialize_base();

private:
   //int uniformWeights(/* PVPatch * wp */ pvdata_t * dataStart, float minwgt, float maxwgt, float sparseFraction, InitUniformRandomWeightsParams *weightParamPtr);
};

} /* namespace PV */
#endif /* INITUNIFORMRANDOMWEIGHTS_HPP_ */
