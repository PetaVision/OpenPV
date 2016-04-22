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
   virtual int communicateInitInfo();
   virtual int allocateDataStructures();
   virtual int updateState(double timef, double dt);
   virtual int setActivity();
protected:
   ShuffleLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_shuffleMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_readFreqFromFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_freqFilename(enum ParamsIOFlag ioFlag);
   virtual void ioParam_freqCollectTime(enum ParamsIOFlag ioFlag);

   void randomShuffle(const pvdata_t * sourceData, pvdata_t * activity);
   void rejectionShuffle(const pvdata_t * sourceData, pvdata_t * activity);
   void collectFreq(const pvdata_t * sourceData);
   void readFreq();

private:
   int initialize_base();
   char * shuffleMethod;
   char * freqFilename;

   long ** featureFreqCount;
   long ** currFeatureFreqCount;

   long * maxCount;
   long freqCollectTime;
   bool readFreqFromFile;
}; // class ShuffleLayer
 
BaseObject * createShuffleLayer(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* ShuffleLayer.hpp */
