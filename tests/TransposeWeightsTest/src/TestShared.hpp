#ifndef TESTSHARED_HPP_
#define TESTSHARED_HPP_

#include <columns/Communicator.hpp>

int TestShared(
      std::string const &testName,
      int nxPre,
      int nyPre,
      int nfPre,
      int nxPost,
      int nyPost,
      int nfPost,
      int patchSizeX,
      int patchSizeY,
      PV::Communicator *comm);

#endif // TESTSHARED_HPP_
