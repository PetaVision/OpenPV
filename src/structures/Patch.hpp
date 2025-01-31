/*
 * Patch
 *
 *  Created on Jul 28, 2017
 *      Author: Pete Schultz
 */

#ifndef PATCH_HPP_
#define PATCH_HPP_

#include <cstdint>

namespace PV {

struct Patch {
   std::uint16_t nx, ny;
   std::uint32_t offset;
};

} // end namespace PV

#endif // PATCH_HPP_
