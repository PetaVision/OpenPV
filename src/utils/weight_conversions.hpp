/*
 * weight_conversions.hpp
 *
 *  Created on: June 18, 2014
 *      Author: Craig Rasmussen
 */

#ifndef WEIGHT_CONVERSIONS_HPP_
#define WEIGHT_CONVERSIONS_HPP_

namespace PV {

/** Compress a weight value to an unsigned char */
static inline uint8_t compressWeight(float w, float minVal, float maxVal) {
   float compressed = 255.0f * ((w - minVal) / (maxVal - minVal)) + 0.5f;
   return static_cast<uint8_t>(compressed);
}

/** Compress a weight value to an unsigned char (weight type float already a uchar) */
static inline uint8_t compressWeight(uint8_t w, float minVal, float maxVal) { return w; }

/** Uncompress a weight value to a float data type */
static inline float uncompressWeight(uint8_t w, float minVal, float maxVal) {
   return (minVal + (maxVal - minVal) * (static_cast<float>(w) / 255.0f));
}

/** Uncompress a weight value to a float data type (weight type float already a float) */
static inline float uncompressWeight(float w, float minVal, float maxVal) { return w; }

} // end namespace PV

#endif /* WEIGHT_CONVERSIONS_HPP_ */
