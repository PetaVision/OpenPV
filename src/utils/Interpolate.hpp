#ifndef INTERPOLATE_HPP_
#define INTERPOLATE_HPP_

#include "structures/Buffer.hpp"

namespace PV {

float interpolate(Buffer<float> const &inputBuffer, float xSrc, float ySrc, int feature);

} // namespace PV

#endif // INTERPOLATE_HPP_
