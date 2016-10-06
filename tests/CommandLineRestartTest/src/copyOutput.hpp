#include "layers/HyPerLayer.hpp"
#include "structures/Buffer.hpp"
#include <vector>

std::vector<pvdata_t> copyOutput(PV::HyPerLayer *layer) {
   PVLayerLoc const *loc = layer->getLayerLoc();
   int const nxExt       = loc->nx + loc->halo.lt + loc->halo.rt;
   int const nyExt       = loc->ny + loc->halo.dn + loc->halo.up;
   int const nf          = loc->nf;
   pvdata_t const *data  = layer->getLayerData();
   PV::Buffer<pvdata_t> outputBuffer{data, nxExt, nyExt, nf};
   return outputBuffer.asVector();
}
