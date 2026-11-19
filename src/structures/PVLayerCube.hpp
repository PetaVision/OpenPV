#ifndef PVLAYERCUBE_HPP_
#define PVLAYERCUBE_HPP_

#include <vector>

#include "structures/PVLayerLoc.hpp"
#include "structures/SparseList.hpp"

namespace PV {

/**
 * PVLayerCube contains pointers to a layer's data, in dense format or sparse format.
 * The width of each vector is the batch width.  
 */
struct PVLayerCube {
   long numItems; // number of items in data buffer
   PVLayerLoc loc;
   bool isExtended;
   bool isSparse;
   std::vector<float const *> data; // pointer to data
   std::vector<SparseList<float> const *> activeIndices;
};

} // namespace PV

#endif // PVLAYERCUBE_HPP_
