#ifndef UTILITYFUNCTIONS_HPP_
#define UTILITYFUNCTIONS_HPP_

#include <columns/Communicator.hpp>
#include <structures/Weights.hpp>

#include <string>

int calcStride(int pre, std::string const &preDesc, int post, std::string const &postDesc);

void checkMPICompatibility(PVLayerLoc const &loc, PV::Communicator const *comm);

PV::Weights createOriginalWeights(
      bool sharedFlag,
      int nxPre,
      int nyPre,
      int nfPre,
      int nxPost,
      int nyPost,
      int nfPost,
      int patchSizeXPre,
      int patchSizeYPre,
      PV::Communicator const *comm);

int checkTransposeOfTranspose(
      std::string const &testName,
      PV::Weights &originalWeights,
      PV::Weights &transposeWeights,
      PV::Communicator const *comm);

#endif // UTILITYFUNCTIONS_HPP_
