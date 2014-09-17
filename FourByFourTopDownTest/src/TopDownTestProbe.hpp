/*
 * TopDownTestProbe.hpp
 *
 *  Created on:
 *      Author: pschultz
 */

#ifndef TOPDOWNTESTPROBE_HPP_
#define TOPDOWNTESTPROBE_HPP_

#include <io/StatsProbe.hpp>
#include <layers/HyPerLayer.hpp>

namespace PV {

class TopDownTestProbe: public PV::StatsProbe {
public:
   TopDownTestProbe(const char * probeName, HyPerCol * hc);
   virtual ~TopDownTestProbe();

   virtual int communicateInitInfo();

   virtual int outputState(double timed);

protected:
   int initTopDownTestProbe(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag);
   virtual void ioParam_checkPeriod(enum ParamsIOFlag ioFlag);
   int setImageLibrary();
   pvdata_t l2distsq(pvdata_t * x, pvdata_t * y);

   int numXPixels;
   int numXGlobal;
   int numYPixels;
   int numYGlobal;
   int numAllPixels;
   int numAllGlobal;
   int numImages;
   int localXOrigin;
   int localYOrigin;
   pvdata_t * imageLibrary;
   pvdata_t * scores;
   double checkperiod;
   double nextupdate;
};

}

#endif /* TOPDOWNTESTPROBE_HPP_ */
