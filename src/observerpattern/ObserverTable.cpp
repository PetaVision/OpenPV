/*
 * ObserverTable.cpp
 *
 *  Created on: Nov 20, 2017
 *      Author: pschultz
 */

#include "observerpattern/ObserverTable.hpp"
#include "utils/PVLog.hpp"

namespace PV {

ObserverTable::ObserverTable(char const *description) { initialize(description); }

ObserverTable::ObserverTable() {}

ObserverTable::~ObserverTable() { clear(); }

void ObserverTable::initialize(char const *description) {
   Observer::initialize();
   setDescription(description);
}

void ObserverTable::addObject(std::string const &name, Observer *entry) {
   mTableAsMultimap.insert(std::make_pair(name, entry));
   mTableAsVector.emplace_back(entry);
}

void ObserverTable::copyTable(ObserverTable const *origTable) {
   FatalIf(
         !mTableAsVector.empty(),
         "copyTable called for %s but the table was not empty.\n",
         getDescription_c());
   auto &multimap = origTable->mTableAsMultimap;
   for (auto &p : multimap) {
      addObject(p.first, p.second);
   }
}

void ObserverTable::clear() {
   mTableAsVector.clear();
   mTableAsMultimap.clear();
}

} // end namespace PV
