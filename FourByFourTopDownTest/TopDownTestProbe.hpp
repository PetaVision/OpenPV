/*
 * TopDownTestProbe.hpp
 *
 *  Created on:
 *      Author: pschultz
 */

#ifndef TOPDOWNTESTPROBE_HPP_
#define TOPDOWNTESTPROBE_HPP_

#include "../PetaVision/src/io/StatsProbe.hpp"
#include "../PetaVision/src/layers/HyPerLayer.hpp"

namespace PV {

class TopDownTestProbe: public PV::StatsProbe {
public:
   TopDownTestProbe(const char * filename, HyPerLayer * layer, const char * msg, float checkperiod);
   virtual ~TopDownTestProbe();

   virtual int outputState(double timed);

protected:
   int initTopDownTestProbe(const char * filename, HyPerLayer * layer, const char * msg, float checkperiod);
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
   float checkperiod;
   float nextupdate;
};

}

#endif /* TOPDOWNTESTPROBE_HPP_ */
