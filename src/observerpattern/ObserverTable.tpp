/*
 * ObserverTable.hpp
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

template <typename S>
S *ObserverTable::lookupByNameRecursive(std::string const &name, int maxIterations) const {
   int n                               = maxIterations;
   ObserverTable const *tableComponent = this;
   S *lookupResult                     = lookupByName<S>(name);
   while (lookupResult == nullptr and n != 0) {
      tableComponent = tableComponent->lookupByType<ObserverTable>();
      if (tableComponent == nullptr) {
         break;
      }
      lookupResult = tableComponent->lookupByName<S>(name);
      n--;
   }
   return lookupResult;
}

template <typename S>
S *ObserverTable::lookupByTypeRecursive(int maxIterations) const {
   int n                               = maxIterations;
   ObserverTable const *tableComponent = this;
   S *lookupResult                     = lookupByType<S>();
   while (lookupResult == nullptr and n != 0) {
      tableComponent = tableComponent->lookupByType<ObserverTable>();
      if (tableComponent == nullptr) {
         break;
      }
      lookupResult = tableComponent->lookupByType<S>();
      n--;
   }
   return lookupResult;
}

} // namespace PV
