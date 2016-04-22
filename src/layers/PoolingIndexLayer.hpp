/*
 * PoolingIndexLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#ifndef POOLINGINDEXLAYER_HPP_
#define POOLINGINDEXLAYER_HPP_

#include "HyPerLayer.hpp"

namespace PV {

class PoolingIndexLayer : public HyPerLayer {
public:
   PoolingIndexLayer(const char* name, HyPerCol * hc);
   virtual ~PoolingIndexLayer();
   virtual int * getActivity(){return (int*)(clayer->activity->data);} // TODO: access to clayer->activity->data should not be public
   virtual int* getChannel(ChannelType ch) {                         // name query
       return (ch < this->numChannels && ch >= 0) ? (int*)GSyn[ch] : NULL;
   }
   bool activityIsSpiking() { return false; }
   virtual int requireChannel(int channelNeeded, int * numChannelsResult);
protected:
   PoolingIndexLayer();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dataType(enum ParamsIOFlag ioFlag);
   virtual int resetGSynBuffers(double timef, double dt);
private:
   int initialize_base();
}; // end of class PoolingIndexLayer

BaseObject * createPoolingIndexLayer(char const * name, HyPerCol * hc);

}  // end namespace PV

#endif /* ANNLAYER_HPP_ */
