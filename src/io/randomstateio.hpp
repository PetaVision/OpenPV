#include "columns/Communicator.hpp"
#include "include/PVLayerLoc.h"
#include "include/pv_types.h"
#include <string>

namespace PV {

double readRandState(
      std::string const &path,
      Communicator *comm,
      taus_uint4 *randState,
      PVLayerLoc const *loc,
      bool extended);

void writeRandState(
      std::string const &path,
      Communicator *comm,
      taus_uint4 const *randState,
      PVLayerLoc const *loc,
      bool extended,
      double simTime,
      bool verifyWrites = false);
}
