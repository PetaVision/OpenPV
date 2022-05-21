#ifndef TESTPVPBATCH_HPP_
#define TESTPVPBATCH_HPP_

#include "structures/MPIBlock.hpp"
#include <memory>
#include <string>

void testPvpBatch(std::shared_ptr<PV::MPIBlock const> mpiBlock, std::string const &directory);

#endif // TESTPVPBATCH_HPP_
