/*
 * InitGaborWeightsParams.hpp
 *
 *  Created on: Aug 13, 2011
 *      Author: kpeterson
 */

#ifndef INITGABORWEIGHTSPARAMS_HPP_
#define INITGABORWEIGHTSPARAMS_HPP_

#include "InitWeightsParams.hpp"

namespace PV {

class InitGaborWeightsParams: public PV::InitWeightsParams {
public:
   InitGaborWeightsParams();
   InitGaborWeightsParams(HyPerConn * parentConn);
   virtual ~InitGaborWeightsParams();
   void calcOtherParams(int patchIndex);

   //get/set methods:
   inline float getaspect()        {return aspect;}
   inline float getshift()        {return shift;}
   inline double getr2Max()        {return r2Max;}
   inline float getsigma()        {return sigma;}
   inline float getlambda()        {return lambda;}
   inline float getphi()        {return phi;}
   inline bool getinvert()        {return invert;}

protected:
   virtual int initialize_base();
   int initialize(HyPerConn * parentConn);

private:
   //params variables:
   float aspect;
   float sigma;
   float rMax;
   double r2Max;
   float strength;
   int lambda;
   float shift;
   //float rotate; // rotate so that axis isn't aligned
   float phi;
   bool invert;
};

} /* namespace PV */
#endif /* INITGABORWEIGHTSPARAMS_HPP_ */
