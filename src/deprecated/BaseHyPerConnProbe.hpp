/*
 * BaseHyPerConnProbe.hpp
 *
 *  Created on: Oct 28, 2014
 *      Author: pschultz
 */

#ifndef BASEHYPERCONNPROBE_HPP_
#define BASEHYPERCONNPROBE_HPP_

#include "BaseConnectionProbe.hpp"
#include "components/Weights.hpp"

namespace PV {

class BaseHyPerConnProbe : public BaseConnectionProbe {
  public:
   BaseHyPerConnProbe(const char *name, PVParams *params, Communicator const *comm);
   virtual ~BaseHyPerConnProbe();

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   // should be const but Weights and PatchGeometry are not const-correct yet
   Weights *getWeights() { return mWeights; }

  protected:
   BaseHyPerConnProbe();
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual void ioParam_statsFlag(enum ParamsIOFlag ioFlag) override;
   virtual bool needRecalc(double timevalue) override;

   /**
    * Implements the referenceUpdateTime method.  Returns the last update time of
    * the target
    * HyPerConn.
    */
   virtual double referenceUpdateTime(double simTime) const override;

   virtual Response::Status outputStateStats(double simTime, double deltaTime) override;

  protected:
   Weights *mWeights; // should be const but Weights and PatchGeometry are not const-correct yet
};

} /* namespace PV */

#endif /* BASEHYPERCONNPROBE_HPP_ */
