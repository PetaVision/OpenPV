/*
 * Patch
 *
 *  Created on Jul 28, 2017
 *      Author: Pete Schultz
 */

#ifndef PVPATCH_HPP_
#define PVPATCH_HPP_

#include <cstdint>

namespace PV {

struct Patch {
   std::uint16_t nx, ny;
   std::uint32_t offset;
};

} // end namespace PV

#endif // PVPATCH_HPP_
