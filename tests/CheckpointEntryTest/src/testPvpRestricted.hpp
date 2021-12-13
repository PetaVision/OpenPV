#ifndef TESTPVPRESTRICTED_HPP_
#define TESTPVPRESTRICTED_HPP_

#include "structures/MPIBlock.hpp"
#include <memory>
#include <string>

void testPvpRestricted(std::shared_ptr<PV::MPIBlock const> mpiBlock, std::string const &directory);

#endif // TESTPVPRESTRICTED_HPP_
