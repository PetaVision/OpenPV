/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef ANNLAYER_HPP_
#define ANNLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class ANNLayer : public HyPerLayer {
public:
   ANNLayer(const char* name, HyPerCol * hc, int numChannels);
   ANNLayer(const char* name, HyPerCol * hc);
   ~ANNLayer();
   virtual int updateV();
   virtual int applyVMax();
   virtual int applyVThresh();
   pvdata_t getVThresh()        { return VThresh; }
   pvdata_t getVMax()           { return VMax; }
   pvdata_t getVMin()           { return VMin; }
protected:
   int initialize();
   virtual int readVThreshParams(PVParams * params);
   pvdata_t VMax;
   pvdata_t VThresh;
   pvdata_t VMin;
}; // end of class ANNLayer

}  // end namespace PV

#endif /* ANNLAYER_HPP_ */
