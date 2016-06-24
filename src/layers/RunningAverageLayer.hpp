/*
 * RunningAverageLayer.hpp
 * This layer stores the running moving average of the most recent n flips (updates) of a layer.
 *
 *  Created on: Mar 3, 2015
 *      Author: wchavez
 */

#ifndef RUNNINGAVERAGELAYER_HPP_
#define RUNNINGAVERAGELAYER_HPP_

#include "CloneVLayer.hpp"

namespace PV {

class RunningAverageLayer: public CloneVLayer {
public:
   RunningAverageLayer(const char * name, HyPerCol * hc);
   virtual ~RunningAverageLayer();
   virtual int communicateInitInfo();
   virtual int allocateV();
   virtual int updateState(double timef, double dt);
   virtual int setActivity();
   int getNumImagesToAverage() { return numImagesToAverage; }

protected:
   RunningAverageLayer();
   int initialize(const char * name, HyPerCol * hc);
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   void ioParam_numImagesToAverage(enum ParamsIOFlag ioFlag);
private:
   int initialize_base();

   // Handled by CloneVLayer
   // char * originalLayerName;
   // HyPerLayer * originalLayer;

protected:
   int numImagesToAverage;
   int numUpdateTimes;
}; // class RunningAverageLayer

BaseObject * createRunningAverageLayer(char const * name, HyPerCol * hc);

}  // namespace PV

#endif /* CLONELAYER_HPP_ */
