/*
 * BaseHyPerConnProbe.hpp
 *
 *  Created on: Oct 28, 2014
 *      Author: pschultz
 */

#ifndef BASEHYPERCONNPROBE_HPP_
#define BASEHYPERCONNPROBE_HPP_

#include "../connections/HyPerConn.hpp"
#include "BaseConnectionProbe.hpp"

namespace PV {

class BaseHyPerConnProbe : public BaseConnectionProbe {
  public:
   BaseHyPerConnProbe(const char *name, HyPerCol *hc);
   virtual ~BaseHyPerConnProbe();

   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   HyPerConn *getTargetHyPerConn() { return dynamic_cast<HyPerConn *>(mTargetConn); }
   HyPerConn const *getTargetHyPerConn() const { return dynamic_cast<HyPerConn *>(mTargetConn); }

  protected:
   BaseHyPerConnProbe();
   int initialize(const char *name, HyPerCol *hc);
   virtual bool needRecalc(double timevalue) override;

   /**
    * Implements the referenceUpdateTime method.  Returns the last update time of
    * the target
    * HyPerConn.
    */
   virtual double referenceUpdateTime() const override;
};

} /* namespace PV */

#endif /* BASEHYPERCONNPROBE_HPP_ */
