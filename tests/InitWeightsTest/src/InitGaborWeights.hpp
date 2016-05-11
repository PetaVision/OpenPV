/*
 * InitGaborWeights.hpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#ifndef INITGABORWEIGHTS_HPP_
#define INITGABORWEIGHTS_HPP_

#include <weightinit/InitWeights.hpp>
#include <weightinit/InitGauss2DWeights.hpp>

namespace PV {

class InitWeightsParams;
class InitGaborWeightsParams;

class InitGaborWeights: public PV::InitGauss2DWeights {
public:
   InitGaborWeights(char const * name, HyPerCol * hc);
   virtual ~InitGaborWeights();

   virtual int calcWeights(pvwdata_t * dataStart, int patchIndex, int arborId);
   virtual InitWeightsParams * createNewWeightParams();
   void calcOtherParams(PVPatch * patch, int patchIndex);


protected:
   InitGaborWeights();
   int initialize(char const * name, HyPerCol * hc);

private:
   int initialize_base();
   int gaborWeights(pvwdata_t * dataStart, InitGaborWeightsParams * weightParamPtr);
};

BaseObject * createInitGaborWeights(char const * name, HyPerCol * hc);

} /* namespace PV */
#endif /* INITGABORWEIGHTS_HPP_ */
