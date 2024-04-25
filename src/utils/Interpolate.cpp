#include "Interpolate.hpp"

namespace PV {

float interpolate(
      Buffer<float> const &inputBuffer, float xSrc, float ySrc, int feature) {
   int const nx = inputBuffer.getWidth();
   int const ny = inputBuffer.getHeight();

   float xSrcFloor = std::floor(xSrc);
   float xSrcInt   = static_cast<int>(xSrcFloor);
   float xSrcFrac  = xSrc - xSrcFloor;

   float ySrcFloor = std::floor(ySrc);
   float ySrcInt   = static_cast<int>(ySrcFloor);
   float ySrcFrac  = ySrc - ySrcFloor;
   
   float valueTL = (xSrcInt >= 0 and xSrcInt < nx and ySrcInt >= 0 and ySrcInt < ny) ?
                   inputBuffer.at(xSrcInt, ySrcInt, feature) : 0.0f;
   valueTL *= (1.0f - xSrcFrac) * (1.0f - ySrcFrac);

   float valueTR = (xSrcInt + 1 >= 0 and xSrcInt + 1 < nx and ySrcInt >= 0 and ySrcInt < ny) ?
                   inputBuffer.at(xSrcInt + 1, ySrcInt, feature) : 0.0f;
   valueTR *= xSrcFrac * (1.0f - ySrcFrac);

   float valueBL = (xSrcInt >= 0 and xSrcInt < nx and ySrcInt + 1 >= 0 and ySrcInt + 1 < ny) ?
                   inputBuffer.at(xSrcInt, ySrcInt + 1, feature) : 0.0f;
   valueBL *= (1.0f - xSrcFrac) * ySrcFrac;

   float valueBR = (xSrcInt + 1 >= 0 and xSrcInt + 1 < nx and ySrcInt + 1 >= 0 and ySrcInt + 1 < ny) ?
                   inputBuffer.at(xSrcInt + 1, ySrcInt + 1, feature) : 0.0f;
   valueBR *= xSrcFrac * ySrcFrac;

   float value = valueTL + valueTR + valueBL + valueBR;
   return value;
}

} // namespace PV
