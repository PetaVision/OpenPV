#ifndef PVPLISTLAYER_HPP__
#define PVPLISTLAYER_HPP__

#include "InputLayer.hpp"

namespace PV {

/**
 * PvpListLayer is an input layer, where InputPath parameter is a file containing a list of
 * .pvp files. The layer reads frame 0 of each .pvp file, choosing the file the same way that
 * ImageLayer chooses a file when its InputPath is a list of files.
 */
class PvpListLayer : public InputLayer {

  public:
   PvpListLayer(char const *name, PVParams *params, Communicator const *comm);
   virtual ~PvpListLayer();

  protected:
   PvpListLayer() {}

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual ActivityComponent *createActivityComponent() override;
};
}

#endif // PVPLISTLAYER_HPP__
