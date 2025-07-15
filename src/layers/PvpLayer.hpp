#ifndef PVPLAYER_HPP__
#define PVPLAYER_HPP__

#include "InputLayer.hpp"

namespace PV {

class PvpLayer : public InputLayer {

  public:
   PvpLayer(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);
   virtual ~PvpLayer();

  protected:
   PvpLayer() {}

   void initialize(std::shared_ptr<ParamsIO> paramsIO, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};

} // end namespace PV

#endif // PVPLAYER_HPP__
