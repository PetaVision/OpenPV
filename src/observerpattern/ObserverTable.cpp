/*
 * ObserverTable.cpp
 *
 *  Created on: Nov 20, 2017
 *      Author: pschultz
 */

#include "observerpattern/ObserverTable.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

ObserverTable::ObserverTable(char const *description) { initialize(description); }

ObserverTable::ObserverTable() {}

ObserverTable::~ObserverTable() { clear(); }

void ObserverTable::initialize(char const *description) {
   Observer::initialize();
   setDescription(description);
}

bool ObserverTable::addObject(std::string const &name, Observer *entry) {
   // auto insertion = mTableAsMap.insert(std::make_pair(std::string(name), entry));
   auto insertion = mTableAsMap.insert(std::make_pair(name, entry));
   // map::insert() returns a pair whose second element is whether the insertion was successful.
   bool addSucceeded = insertion.second;
   if (addSucceeded) {
      mTableAsVector.emplace_back(entry);
   }
   return addSucceeded;
}

void ObserverTable::copyTable(ObserverTable const *origTable) {
   FatalIf(
         !mTableAsVector.empty(),
         "copyTable called for %s but the table was not empty.\n",
         getDescription_c());
   auto &map = origTable->mTableAsMap;
   for (auto &p : map) {
      addObject(p.first, p.second);
   }
}

void ObserverTable::clear() {
   mTableAsVector.clear();
   mTableAsMap.clear();
}

} // end namespace PV
