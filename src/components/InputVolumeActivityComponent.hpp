/*
 * InputVolumeActivityComponent.hpp
 */

#ifndef INPUTVOLUMEACTIVITYCOMPONENT_HPP_
#define INPUTVOLUMEACTIVITYCOMPONENT_HPP_

#include "components/ActivityComponent.hpp"

namespace PV {

class InputVolumeActivityComponent : public ActivityComponent {
  public:
   InputVolumeActivityComponent(char const *name, PVParams *params, Communicator const *comm);

   virtual ~InputVolumeActivityComponent() {}

  protected:
   InputVolumeActivityComponent() {}

   virtual ActivityBuffer *createActivity() override;

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual void setObjectType() override;
}; // class InputVolumeActivityComponent

} // namespace PV

#endif // INPUTVOLUMEACTIVITYCOMPONENT_HPP_
