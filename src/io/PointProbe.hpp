/*
 * PointProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: rasmussn
 */

#ifndef POINTPROBE_HPP_
#define POINTPROBE_HPP_

#include "LayerProbe.hpp"

namespace PV {

class PointProbe: public PV::LayerProbe {
public:
   PointProbe(const char * probeName, HyPerCol * hc);
   virtual ~PointProbe();

   virtual int communicateInitInfo();

   virtual int outputState(double timef);

protected:
   int xLoc;
   int yLoc;
   int fLoc;
   int batchLoc;

   PointProbe();
   int initialize(const char * probeName, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_xLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_yLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_fLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_batchLoc(enum ParamsIOFlag ioFlag);
   virtual int initOutputStream(const char * filename);
   virtual int writeState(double timef);
   
   /**
    * Overrides initNumValues() to set numValues to 2 (membrane potential and activity)
    */
   virtual int initNumValues();
   
   /**
    * Implements calcValues for PointProbe.  probeValues[0] is the point's membrane potential and probeValues[1] is the point's activity.
    * If the target layer does not have a membrane potential, probeValues[0] is zero.
    * Note that under MPI, only the root process and the process containing the neuron being probed contain
    * the values.
    */
   virtual int calcValues(double timevalue);
   

private:
   int initPointProbe_base();
   
   /**
    * A convenience method to return probeValues[0] (the membrane potential).  Note that it does not call needRecalc().
    */
   inline double getV();
   
   /**
    * A convenience method to return probeValues[0] (the activity).  Note that it does not call needRecalc().
    */   
   inline double getA();
}; // end class PointProbe

BaseObject * createPointProbe(char const * name, HyPerCol * hc);

}

#endif /* POINTPROBE_HPP_ */
