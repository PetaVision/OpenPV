/*
 * LIFGap.hpp
 *
 *  Created on: Jul 29, 2011
 *      Author: garkenyon
 */

#ifndef LIFGAP_HPP_
#define LIFGAP_HPP_

#include "LIF.hpp"

#define NUM_LIFGAP_EVENTS 1 + NUM_LIF_EVENTS // ???
//#define EV_LIF_GSYN_GAP     NUM_LIF_EVENTS + 1
#define EV_LIFGAP_GSYN_GAP 3
//#define EV_LIFGAP_ACTIVITY  4

namespace PV {

class LIFGap : public PV::LIF {
  public:
   LIFGap(const char *name, HyPerCol *hc);
   virtual ~LIFGap();

   virtual Response::Status updateState(double time, double dt) override;

   virtual Response::Status readStateFromCheckpoint(Checkpointer *checkpointer) override;

   const float *getGapStrength() { return gapStrength; }

  protected:
   LIFGap();
   int initialize(const char *name, HyPerCol *hc, const char *kernel_name);
   virtual void allocateConductances(int num_channels) override;
   virtual Response::Status
   registerData(std::shared_ptr<RegisterDataMessage<Checkpointer> const> message) override;
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;
   virtual void readGapStrengthFromCheckpoint(Checkpointer *checkpointer);

  private:
   int initialize_base();
   void calcGapStrength();

   float *gapStrength          = nullptr;
   bool gapStrengthInitialized = false;

}; // class LIFGap

} /* namespace PV */
#endif /* LIFGAP_HPP_ */
