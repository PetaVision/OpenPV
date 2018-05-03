/*
 * MapLookup.hpp
 *
 *  Created on: Nov 23, 2017
 *      Author: Pete Schultz
 */

#ifndef MAPLOOKUPBYTYPE_HPP_
#define MAPLOOKUPBYTYPE_HPP_

#include "observerpattern/Observer.hpp"

namespace PV {

/**
 * Given a map with strings for keys and pointers to Observers for values,
 * this function template looks for a pointer to an observer that casts
 * to the template parameter. If there is none, it returns the null pointer.
 * If there is more than one, it exits with an error.
 */
template <typename S>
S *mapLookupByType(std::map<std::string, Observer *> const &objectMap, std::string const &ident) {
   // TODO: should be possible to do this very compactly using <algorithm>
   S *foundObject = nullptr;
   for (auto &objpair : objectMap) {
      auto *obj       = objpair.second;
      auto castObject = dynamic_cast<S *>(obj);
      if (castObject != nullptr) {
         FatalIf(
               foundObject != nullptr,
               "mapLookupByType found more than one object of the specified type in %s.\n",
               ident.c_str());
         foundObject = castObject;
      }
   }

   return foundObject;
} // mapLookupByType

} // namespace PV

#endif // MAPLOOKUPBYTYPE_HPP_
