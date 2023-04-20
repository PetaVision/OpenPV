/*
 * DatastoreDelayTestProbe.hpp
 *
 *  Created on: Mar 10, 2009
 *      Author: garkenyon
 */

#ifndef DATASTOREDELAYTESTPROBE_HPP_
#define DATASTOREDELAYTESTPROBE_HPP_

#include "columns/Communicator.hpp"
#include "columns/Messages.hpp"
#include "components/BasePublisherComponent.hpp"
#include "io/PVParams.hpp"
#include "observerpattern/Response.hpp"
#include "probes/StatsProbeImmediate.hpp"
#include <memory>

namespace PV {

class DatastoreDelayTestProbe : public StatsProbeImmediate {
  public:
   DatastoreDelayTestProbe(const char *name, PVParams *params, Communicator const *comm);

   virtual ~DatastoreDelayTestProbe();

  protected:
   virtual void checkStats() override;
   virtual void createProbeLocal(char const *name, PVParams *params) override;
   void initialize(const char *name, PVParams *params, Communicator const *comm);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;

   // Data members
  private:
   BasePublisherComponent *mInputPublisher = nullptr;
};

} // namespace PV

#endif /* DATASTOREDELAYTESTPROBE_HPP_ */
