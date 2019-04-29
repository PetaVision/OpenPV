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

template <typename T>
T *ObserverTable::findObject(std::string const &name) const {
   std::vector<T *> matches = findObjects<T>(name);
   if (matches.empty()) {
      return nullptr;
   }
   FatalIf(
         matches.size() > (std::size_t)1,
         "findObject found more than one object of matching type with name \"%s\".\n",
         name.c_str());
   return matches[0];
}

template <typename T>
T *ObserverTable::findObject(char const *name) const {
   return findObject<T>(std::string(name));
}

template <typename T>
std::vector<T *> ObserverTable::findObjects(std::string const &name) const {
   auto matches = mTableAsMultimap.equal_range(name);
   std::vector<T *> result;
   result.reserve(std::distance(matches.first, matches.second));
   for (auto &match = matches.first; match != matches.second; match++) {
      T *castObject = dynamic_cast<T *>(match->second);
      if (castObject) {
         result.emplace_back(castObject);
      }
   }
   return result;
}

template <typename T>
std::vector<T *> ObserverTable::findObjects(char const *name) const {
   return findObjects<T>(std::string(name));
}

} // namespace PV
