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

class PointProbe : public PV::LayerProbe {
  public:
   PointProbe(const char *name, HyPerCol *hc);
   virtual ~PointProbe();

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual int outputState(double timef) override;

  protected:
   int xLoc;
   int yLoc;
   int fLoc;
   int batchLoc;

   PointProbe();
   int initialize(const char *name, HyPerCol *hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_xLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_yLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_fLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_batchLoc(enum ParamsIOFlag ioFlag);

   /**
    * Overrides initOutputStreams. A process whose restricted region contains
    * the indicated point has an mOutputStreams vector whose length is the
    * local batch width. Other processes have an empty mOutputStreams vector.
    */
   virtual void initOutputStreams(const char *filename, Checkpointer *checkpointer) override;

   virtual int writeState(double timevalue);

   /**
    * Overrides initNumValues() to set numValues to 2 (membrane potential and
    * activity)
    */
   virtual int initNumValues() override;

   /**
    * Implements calcValues for PointProbe.  probeValues[0] is the point's
    * membrane potential and
    * probeValues[1] is the point's activity.
    * If the target layer does not have a membrane potential, probeValues[0] is
    * zero.
    * Note that under MPI, only the root process and the process containing the
    * neuron being probed
    * contain
    * the values.
    */
   virtual int calcValues(double timevalue) override;

  private:
   int initialize_base();

   /**
    * A convenience method to return probeValues[0] (the membrane potential).
    * Note that it does not
    * call needRecalc().
    */
   inline double getV();

   /**
    * A convenience method to return probeValues[0] (the activity).  Note that it
    * does not call
    * needRecalc().
    */
   inline double getA();
}; // end class PointProbe
}

#endif /* POINTPROBE_HPP_ */
