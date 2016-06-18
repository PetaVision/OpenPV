/*
 * LocalizationData.hpp
 *
 *  Created on: May 23, 2016
 *      Author: pschultz
 */

#ifndef LOCALIZATIONDATA_HPP_
#define LOCALIZATIONDATA_HPP_

/**
 * A struct to contain a bounding box, its feature, its displayed category index, and a score reflecting confidence of the bounding box detection.
 */
struct LocalizationData {
   int feature;
   int displayedIndex;
   int left;
   int right;
   int top;
   int bottom;
   double score;
};

#endif /* LOCALIZATIONDATA_HPP_ */
