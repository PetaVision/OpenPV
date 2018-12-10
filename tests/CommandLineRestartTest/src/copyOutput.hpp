#include "components/PublisherComponent.hpp"
#include "layers/HyPerLayer.hpp"
#include "structures/Buffer.hpp"
#include <vector>

std::vector<float> copyOutput(PV::HyPerLayer *layer) {
   auto *publisher       = layer->getComponentByType<PV::PublisherComponent>();
   PVLayerLoc const *loc = publisher->getLayerLoc();
   int const nxExt       = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExt       = loc->ny + loc->halo.dn + loc->halo.up;
   int const nf          = loc->nf;
   float const *data     = publisher->getLayerData();
   PV::Buffer<float> outputBuffer{data, nxExt, nyExt, nf};
   return outputBuffer.asVector();
}
