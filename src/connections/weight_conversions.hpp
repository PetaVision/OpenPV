/*
 * weight_conversions.hpp
 *
 *  Created on: June 18, 2014
 *      Author: Craig Rasmussen
 */

#ifndef WEIGHT_CONVERSIONS_H_
#define WEIGHT_CONVERSIONS_H_

namespace PV {

/** Compress a weight value to an unsigned char */
static inline unsigned char compressWeight(float w, float minVal, float maxVal) {
   return (unsigned char)(255.0f * ((w - minVal) / (maxVal - minVal)) + 0.5f);
}

/** Compress a weight value to an unsigned char (weight type float already a uchar) */
static inline unsigned char compressWeight(unsigned char w, float minVal, float maxVal) {
   return w;
}

/** Uncompress a weight value to a float data type */
static inline float uncompressWeight(unsigned char w, float minVal, float maxVal) {
   return (float)(minVal + (maxVal - minVal) * ((float)w / 255.0f));
}

/** Uncompress a weight value to a float data type (weight type float already a float) */
static inline float uncompressWeight(float w, float minVal, float maxVal) { return w; }

} // end namespace PV

#endif /* WEIGHT_CONVERSIONS_H_ */
