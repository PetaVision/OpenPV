/*
 * SUPointProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#ifndef SUPOINTPROBE_HPP_
#define SUPOINTPROBE_HPP_

#include <io/PointProbe.hpp>
#include <layers/Movie.hpp>

namespace PV {

class SUPointProbe: public PV::PointProbe{
public:
   SUPointProbe(const char * probeName, HyPerCol * hc);
   virtual ~SUPointProbe();

   virtual int communicateInitInfo();

   virtual int outputState(double timef);

protected:

   SUPointProbe();
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_xLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_yLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_fLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_disparityLayerName(enum ParamsIOFlag ioFlag);
   virtual int point_writeState(double timef, float outVVal, float outAVal); 

private:
   int initSUPointProbe_base();
   char * disparityLayerName;
   Movie* disparityLayer;
};

}

#endif /* POINTPROBE_HPP_ */
