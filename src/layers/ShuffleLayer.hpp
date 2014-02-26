/*
 * ShuffleLayer.hpp
 * Used to shuffle active features
 * to build a psychophysical mask
 *
 *  Created: July, 2013
 *   Author: Sheng Lundquist, Will Shainin
 */

#ifndef SHUFFLELAYER_HPP_
#define SHUFFLELAYER_HPP_

#include "CloneVLayer.hpp"

namespace PV {

class ShuffleLayer: public CloneVLayer {
public:
   ShuffleLayer(const char * name, HyPerCol * hc);
   virtual ~ShuffleLayer();
   virtual int updateState(double timef, double dt);
   virtual int setActivity();
protected:
   ShuffleLayer();
   int initialize(const char * name, HyPerCol * hc);
   int communicateInitInfo();

   void randomShuffle(const pvdata_t * sourceData, pvdata_t * activity);
   void rejectionShuffle(const pvdata_t * sourceData, pvdata_t * activity);
   void collectFreq(const pvdata_t * sourceData);

   virtual int setParams(PVParams * params);
   void readShuffleMethod(PVParams * params);
   void readFreqCollectTime(PVParams * params);

private:
   int initialize_base();
   char * shuffleMethod;
   long * featureFreqCount;
   long maxCount;
   long freqCollectTime;
};

}

#endif /* ShuffleLayer.hpp */
