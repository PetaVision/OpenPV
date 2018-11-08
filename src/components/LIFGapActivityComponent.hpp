/*
 * LIFGapActivityComponent.hpp
 *
 *  Created on: Jul 29, 2011
 *      Author: garkenyon
 */

#ifndef LIFGAPACTIVITYCOMPONENT_HPP_
#define LIFGAPACTIVITYCOMPONENT_HPP_

#include "LIFActivityComponent.hpp"

namespace PV {

class LIFGapActivityComponent : public LIFActivityComponent {
  public:
   LIFGapActivityComponent(const char *name, PVParams *params, Communicator *comm);
   virtual ~LIFGapActivityComponent();

   virtual Response::Status updateActivity(double simTime, double deltaTime) override;

   float const *getGapStrength() { return mGapStrength->getBufferData(); }

  protected:
   LIFGapActivityComponent();
   void initialize(const char *name, PVParams *params, Communicator *comm);
   virtual void createComponentTable(char const *tableDescription) override;
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   virtual Response::Status
   initializeState(std::shared_ptr<InitializeStateMessage const> message) override;

   void updateActivityOriginal(
         int const nbatch,
         int const numNeurons,
         float const simTime,
         float const deltaTime,

         int const nx,
         int const ny,
         int const nf,
         int const lt,
         int const rt,
         int const dn,
         int const up,

         LIFParams *params,
         taus_uint4 *rnd,

         float *V,
         float *Vth,
         float *G_E,
         float *G_I,
         float *G_IB,
         float const *GSynHead,
         float *activity,

         float const *gapStrength);

   void updateActivityBeginning(
         int const nbatch,
         int const numNeurons,
         float const simTime,
         float const deltaTime,

         int const nx,
         int const ny,
         int const nf,
         int const lt,
         int const rt,
         int const dn,
         int const up,

         LIFParams *params,
         taus_uint4 *rnd,

         float *V,
         float *Vth,
         float *G_E,
         float *G_I,
         float *G_IB,
         float const *GSynHead,
         float *activity,

         float const *gapStrength);

   void updateActivityArma(
         int const nbatch,
         int const numNeurons,
         float const simTime,
         float const deltaTime,

         int const nx,
         int const ny,
         int const nf,
         int const lt,
         int const rt,
         int const dn,
         int const up,

         LIFParams *params,
         taus_uint4 *rnd,

         float *V,
         float *Vth,
         float *G_E,
         float *G_I,
         float *G_IB,
         float const *GSynHead,
         float *activity,

         float const *gapStrength);

  private:
   void calcGapStrength();

   RestrictedBuffer *mGapStrength = nullptr;
   bool mGapStrengthInitialized   = false;

}; // class LIFGapActivityComponent

} /* namespace PV */
#endif /* LIFGAPACTIVITYCOMPONENT_HPP_ */
