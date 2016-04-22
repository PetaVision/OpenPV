/*
 * InitUniformWeightsParams.hpp
 *
 *  Created on: Aug 23, 2011
 *      Author: kpeterson
 */

#ifndef INITUNIFORMWEIGHTSPARAMS_HPP_
#define INITUNIFORMWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitUniformWeightsParams: public PV::InitWeightsParams {
public:
   InitUniformWeightsParams();
   InitUniformWeightsParams(const char * name, HyPerCol * hc);
   virtual ~InitUniformWeightsParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   //get-set methods:
   inline float getInitWeight()        {return initWeight;}
   inline bool getConnectOnlySameFeatures()        {return connectOnlySameFeatures;}

protected:
   virtual int initialize_base();
   int initialize(const char * name, HyPerCol * hc);
   virtual void ioParam_weightInit(enum ParamsIOFlag ioFlag);
   virtual void ioParam_connectOnlySameFeatures(enum ParamsIOFlag ioFlag);

private:
   float initWeight;
   bool connectOnlySameFeatures;

};

} /* namespace PV */
#endif /* INITUNIFORMWEIGHTSPARAMS_HPP_ */
