#include "include/PVLayerLoc.h"
#include "include/pv_types.h"
#include "structures/MPIBlock.hpp"
#include <string>

namespace PV {

double readRandState(
      std::string const &path,
      MPIBlock const *mpiBlock,
      taus_uint4 *randState,
      PVLayerLoc const *loc,
      bool extended);

void writeRandState(
      std::string const &path,
      MPIBlock const *mpiBlock,
      taus_uint4 const *randState,
      PVLayerLoc const *loc,
      bool extended,
      double simTime,
      bool verifyWrites = false);
}
