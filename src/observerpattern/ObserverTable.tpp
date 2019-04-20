/*
 * ObserverTable.tpp
 *
 *
 *  Created on Oct 3, 2018
 *      Author: Pete Schultz
 *  method template implementations for the ObserverTable class template.
 *  Note that the .hpp includes this .tpp file at the end;
 *  the .tpp file does not include the .hpp file.
 */

namespace PV {

// Note: the type T must have a getName() function member, but the Observer class doesn't.
template <typename T>
T *ObserverTable::findObject(std::string const &name) const {
   T *result = nullptr;
   for (auto iterator = begin(); iterator != end(); iterator++) {
      T *castObject = dynamic_cast<T *>(*iterator);
      if (castObject and name == castObject->getName()) {
         FatalIf(
               result,
               "findObject found more than one object of matching type with name \"%s\".\n",
               name.c_str());
         result = castObject;
      }
   }
   return result;
}

template <typename T>
std::vector<T *> ObserverTable::findObjects(std::string const &name) const {
   auto matches = mTableAsMultimap.equal_range(name);
   std::vector<T *> result;
   result.reserve(std::distance(matches.first, matches.second));
   for (auto &match = matches.first; match != matches.second; match++) {
      T *castObject = dynamic_cast<T *>(match->second);
      if (castObject) {
         // pvAssert(name == castObject->getName());
         result.emplace_back(castObject);
      }
   }
   return result;
}

template <typename T>
T *ObserverTable::findObject(char const *name) const {
   return findObject<T>(std::string(name));
}

} // namespace PV
