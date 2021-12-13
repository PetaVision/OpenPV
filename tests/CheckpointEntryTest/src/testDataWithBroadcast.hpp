#ifndef TESTDATAWITHBROADCAST_HPP_
#define TESTDATAWITHBROADCAST_HPP_

#include "structures/MPIBlock.hpp"
#include <memory>
#include <string>

void testDataWithBroadcast(std::shared_ptr<PV::MPIBlock const> mpiBlock, std::string const &directory);

#endif // TESTDATAWITHBROADCAST_HPP_
