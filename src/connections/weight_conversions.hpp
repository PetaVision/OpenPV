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
static inline unsigned char compressWeight(pvdata_t w, pvdata_t minVal, pvdata_t maxVal) {
   return (unsigned char) (255.0f * ((w - minVal) / (maxVal - minVal)) + 0.5f);
}

/** Compress a weight value to an unsigned char (weight type pvwdata_t already a uchar) */
static inline unsigned char compressWeight(unsigned char w, pvdata_t minVal, pvdata_t maxVal) {
   return w;
}

/** Uncompress a weight value to a pvdata_t data type */
static inline pvdata_t uncompressWeight(unsigned char w, pvdata_t minVal, pvdata_t maxVal) {
   return (pvdata_t) (minVal + (maxVal - minVal) * ((pvdata_t)w / 255.0f));
}

/** Uncompress a weight value to a pvdata_t data type (weight type pvwdata_t already a float) */
static inline pvdata_t uncompressWeight(float w, pvdata_t minVal, pvdata_t maxVal) {
   return w;
}

}  // end namespace PV

#endif /* WEIGHT_CONVERSIONS_H_ */
