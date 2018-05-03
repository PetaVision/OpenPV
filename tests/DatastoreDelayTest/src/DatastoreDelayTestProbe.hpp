/*
 * DatastoreDelayTestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef DATASTOREDELAYTESTPROBE_HPP_
#define DATASTOREDELAYTESTPROBE_HPP_

#include "columns/HyPerCol.hpp"
#include "include/pv_common.h"
#include "probes/StatsProbe.hpp"

namespace PV {

class DatastoreDelayTestProbe : public StatsProbe {
  protected:
   virtual void ioParam_buffer(enum ParamsIOFlag ioFlag) override;

  public:
   DatastoreDelayTestProbe(const char *name, HyPerCol *hc);

   virtual Response::Status outputState(double timestamp) override;

   virtual ~DatastoreDelayTestProbe();

  protected:
   int initialize(const char *name, HyPerCol *hc);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   // Data members
  private:
   int mNumDelayLevels = 0;
};
}

#endif /* DATASTOREDELAYTESTPROBE_HPP_ */
