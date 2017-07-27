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

class ShuffleLayer : public CloneVLayer {
  public:
   ShuffleLayer(const char *name, HyPerCol *hc);
   virtual ~ShuffleLayer();
   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual int allocateDataStructures() override;
   virtual int updateState(double timef, double dt) override;
   virtual int setActivity() override;

  protected:
   ShuffleLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_shuffleMethod(enum ParamsIOFlag ioFlag);
   virtual void ioParam_readFreqFromFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_freqFilename(enum ParamsIOFlag ioFlag);
   virtual void ioParam_freqCollectTime(enum ParamsIOFlag ioFlag);

   void randomShuffle(const float *sourceData, float *activity);
   void rejectionShuffle(const float *sourceData, float *activity);
   void collectFreq(const float *sourceData);
   void readFreq();

  private:
   int initialize_base();
   char *shuffleMethod;
   char *freqFilename;

   long **featureFreqCount;
   long **currFeatureFreqCount;

   long *maxCount;
   long freqCollectTime;
   bool readFreqFromFile;
}; // class ShuffleLayer

} // namespace PV

#endif /* ShuffleLayer.hpp */
