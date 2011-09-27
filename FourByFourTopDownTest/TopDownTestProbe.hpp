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
   TopDownTestProbe(const char * filename, HyPerCol * hc, const char * msg, float checkperiod);
   virtual ~TopDownTestProbe();

   virtual int outputState(float time, HyPerLayer * l);

protected:
   int initialize(float checkperiod);
   int setImageLibrary();
   int resetImageLibrary(HyPerLayer * l);
   pvdata_t l2distsq(pvdata_t * x, pvdata_t * y);

   HyPerLayer * prev_l;
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
