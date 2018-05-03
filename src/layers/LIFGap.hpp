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
   virtual Response::Status registerData(Checkpointer *checkpointer) override;
   virtual void readGapStrengthFromCheckpoint(Checkpointer *checkpointer);

  private:
   int initialize_base();
   float *gapStrength;
   bool gapStrengthInitialized;
   void calcGapStrength();

}; // class LIFGap

} /* namespace PV */
#endif /* LIFGAP_HPP_ */
