/*
 * InitBIDSLateralParams.hpp
 *
 *  Created on: Aug 10, 2012
 *      Author: bnowers
 */

#ifndef INITBIDSLATERALPARAMS_HPP_
#define INITBIDSLATERALPARAMS_HPP_

#include <weightinit/InitWeightsParams.hpp>
#include "BIDSLayer.hpp"
#include "BIDSMovieCloneMap.hpp"

namespace PV {

class InitBIDSLateralParams: public PV::InitWeightsParams {
public:
   InitBIDSLateralParams();
   InitBIDSLateralParams(const char * name, HyPerCol * hc);
   virtual ~InitBIDSLateralParams();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual int communicateParamsInfo();
   void calcOtherParams(int patchIndex);

   //get-set methods:
   inline BIDSCoords *getCoords()  {return coords;}
   inline const char * getFalloffType()  {return falloffType;}
   inline const char * getJitterSource() {return jitterSource;}
   inline int getLateralRadius()   {return lateralRadius;}
   inline float getStrength() {return strength;}
   inline int getJitter() {return jitter;}

protected:
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc);
   virtual void ioParam_strength(enum ParamsIOFlag ioFlag);
   virtual void ioParam_falloffType(enum ParamsIOFlag ioFlag);
   virtual void ioParam_lateralRadius(enum ParamsIOFlag ioFlag);
   virtual void ioParam_jitterSource(enum ParamsIOFlag ioFlag);

private:

   //params file values:
   float strength;
   BIDSCoords * coords; //structure array pointer that holds the randomly generated corrdinates for the specified number of BIDS nodes
   char * falloffType;
   char * jitterSource;
   int lateralRadius;
   int jitter;
};

} /* namespace PV */
#endif /* INITBIDSLATERALPARAMS_HPP_ */
