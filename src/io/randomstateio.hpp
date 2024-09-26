#ifndef RANDOMSTATEIO_HPP_
#define RANDOMSTATEIO_HPP_

#include "include/PVLayerLoc.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/cl_random.h"
#include <memory>
#include <string>

namespace PV {

double readRandState(
      std::string const &path,
      std::shared_ptr<MPIBlock const> mpiBlock,
      taus_uint4 *randState,
      PVLayerLoc const *loc,
      bool extended);

void writeRandState(
      std::string const &path,
      std::shared_ptr<MPIBlock const> mpiBlock,
      taus_uint4 const *randState,
      PVLayerLoc const *loc,
      bool extended,
      double simTime,
      bool verifyWrites = false);
}

#endif // RANDOMSTATEIO_HPP_
