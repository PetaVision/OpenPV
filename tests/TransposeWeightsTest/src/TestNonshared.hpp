#ifndef TESTNONSHARED_HPP_
#define TESTNONSHARED_HPP_

#include <columns/Communicator.hpp>

int TestNonshared(
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

#endif // TESTNONSHARED_HPP_
