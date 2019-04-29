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

class PointProbe : public LayerProbe {
  public:
   PointProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~PointProbe();

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   virtual Response::Status outputState(double simTime, double deltaTime) override;

   float const *getPointV() const { return mPointV; }
   float const *getPointA() const { return mPointA; }
   int getPointRank() const { return mPointRank; }

  protected:
   int xLoc;
   int yLoc;
   int fLoc;
   int batchLoc;

   PointProbe();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   virtual void ioParam_xLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_yLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_fLoc(enum ParamsIOFlag ioFlag);
   virtual void ioParam_batchLoc(enum ParamsIOFlag ioFlag);

   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   /**
    * Overrides initOutputStreams. A process whose restricted region contains
    * the indicated point has an mOutputStreams vector whose length is the
    * local batch width. Other processes have an empty mOutputStreams vector.
    */
   virtual void initOutputStreams(const char *filename, Checkpointer *checkpointer) override;

   virtual void writeState(double timevalue);

   /**
    * Overrides initNumValues() to set numValues to 2 (membrane potential and
    * activity)
    */
   virtual void initNumValues() override;

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
   virtual void calcValues(double timevalue) override;

  private:
   int initialize_base();

  private:
   int mPointRank = 0; // The global rank for which the target point is in the restricted region.
   // mPointV, mPointA, and mPointRank are set in InitializeState.
   float const *mPointV = nullptr; // Points to the memory location of the target point's V
   float const *mPointA = nullptr; // Points to the memory location of the target point's A
}; // end class PointProbe
}

#endif /* POINTPROBE_HPP_ */
