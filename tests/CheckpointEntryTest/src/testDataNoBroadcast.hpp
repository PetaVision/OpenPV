#ifndef TESTDATANOBROADCAST_HPP_
#define TESTDATANOBROADCAST_HPP_

#include "structures/MPIBlock.hpp"
#include <memory>
#include <string>

void testDataNoBroadcast(std::shared_ptr<PV::MPIBlock const> mpiBlock, std::string const &directory);

#endif // TESTDATANOBROADCAST_HPP_
