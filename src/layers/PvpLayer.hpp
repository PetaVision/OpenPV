#ifndef PVPLAYER_HPP__
#define PVPLAYER_HPP__

#include "InputLayer.hpp"

namespace PV {

class PvpLayer : public InputLayer {

  public:
   PvpLayer(char const *name, HyPerCol *hc);
   virtual ~PvpLayer();

  protected:
   PvpLayer() {}

   int initialize(char const *name, HyPerCol *hc);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // PVPLAYER_HPP__
