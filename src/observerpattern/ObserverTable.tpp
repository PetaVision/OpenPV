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

template <typename S>
S *ObserverTable::lookupByName(std::string const &name) const {
   S *lookupResult = nullptr;
   auto findResult = mTableAsMap.find(name);
   if (findResult != mTableAsMap.end()) {
      auto observerPtr = findResult->second;
      lookupResult     = dynamic_cast<S *>(observerPtr);
   }
   return lookupResult;
}

template <typename S>
S *ObserverTable::lookupByType() const {
   S *lookupResult = nullptr;
   for (auto *obs : mTableAsVector) {
      auto castObject = dynamic_cast<S *>(obs);
      if (castObject) {
         FatalIf(
               lookupResult,
               "lookupByType called but %s has multiple objects of the given type.\n",
               getDescription_c());
         lookupResult = castObject;
      }
   }
   return lookupResult;
}

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
T *ObserverTable::findObject(char const *name) const {
   return findObject<T>(std::string(name));
}

} // namespace PV
