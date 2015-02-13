/*
 * InitUnivormRandomWeightsParams.hpp
 *
 *  Created on: Aug 12, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIVORMRANDOMWEIGHTSPARAMS_HPP_
#define INITUNIVORMRANDOMWEIGHTSPARAMS_HPP_

#include "InitRandomWeightsParams.hpp"

namespace PV {

class InitUniformRandomWeightsParams: public PV::InitRandomWeightsParams {
public:
   InitUniformRandomWeightsParams();
   InitUniformRandomWeightsParams(const char * name, HyPerCol * hc);
   virtual ~InitUniformRandomWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   //get-set methods:
   inline float getWMin()        {return wMin;}
   inline float getWMax()        {return wMax;}
   inline float getSparseFraction()        {return sparseFraction;}

protected:
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc);
   virtual void ioParam_wMinInit(enum ParamsIOFlag ioFlag);
   virtual void ioParam_wMaxInit(enum ParamsIOFlag ioFlag);
   virtual void ioParam_sparseFraction(enum ParamsIOFlag ioFlag);


private:
   float wMin;
   float wMax;
   float sparseFraction;  // fraction of weights identically zero:  0 (default) -> no sparseness, 1 -> all weights == 0
};

} /* namespace PV */
#endif /* INITUNIVORMRANDOMWEIGHTSPARAMS_HPP_ */
