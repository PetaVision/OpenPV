#ifndef PVPLAYER_HPP__
#define PVPLAYER_HPP__

#include "InputLayer.hpp"

namespace PV {

class PvpLayer : public InputLayer {

  public:
   PvpLayer(char const *name, PVParams *params, Communicator *comm);
   virtual ~PvpLayer();

  protected:
   PvpLayer() {}

   void initialize(char const *name, PVParams *params, Communicator *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // PVPLAYER_HPP__
