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

   virtual int
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   HyPerConn *getTargetHyPerConn() { return targetHyPerConn; }

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

  private:
   int initialize_base();

  protected:
   HyPerConn *targetHyPerConn;
};

} /* namespace PV */

#endif /* BASEHYPERCONNPROBE_HPP_ */
