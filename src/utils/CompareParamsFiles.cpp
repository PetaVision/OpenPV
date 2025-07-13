/*
 * CompareParamsFiles.cpp
 *
 *  Created on: Dec 17, 2018
 *      Author: pschultz
 */

#include "CompareParamsFiles.hpp"
#include "include/pv_common.h"

namespace PV {

int compareParamsFiles(
      std::string const &paramsFile1,
      std::string const &paramsFile2,
      MPI_Comm mpiComm) {
   int status = PV_SUCCESS;
   PVParams params1{paramsFile1.c_str(), mpiComm};
   PVParams params2{paramsFile2.c_str(), mpiComm};

   // create a map between groups in paramsFile1 and those in paramsFile2.
   std::map<std::shared_ptr<ParamGroup>, std::shared_ptr<ParamGroup>> paramGroupMap;
   char const *groupName = nullptr;
   for (int idx = 0; (groupName = params1.groupNameFromIndex(idx)) != nullptr; idx++) {
      std::shared_ptr<ParamGroup> g1 = params1.group(groupName);
      std::shared_ptr<ParamGroup> g2 = params2.group(groupName);
      if (g2 == nullptr) {
         ErrorLog().printf(
               "Group name \"%s\" is in \"%s\" but not in \"%s\".\n",
               groupName,
               paramsFile1.c_str(),
               paramsFile2.c_str());
         status = PV_FAILURE;
      }
      else {
         paramGroupMap.emplace(std::make_pair(g1, g2));
      }
   }
   for (int idx = 0; (groupName = params2.groupNameFromIndex(idx)) != nullptr; idx++) {
      if (params1.group(groupName) == nullptr) {
         ErrorLog().printf(
               "Group name \"%s\" is in \"%s\" but not in \"%s\".\n",
               groupName,
               paramsFile2.c_str(),
               paramsFile1.c_str());
         status = PV_FAILURE;
      }
   }

   for (auto &p : paramGroupMap) {
      status |= compareParamGroups(p.first, p.second);
   }
   return status;
}

int compareParamGroups(std::shared_ptr<ParamGroup> group1, std::shared_ptr<ParamGroup> group2) {
   int status = PV_SUCCESS;
   if (group1->getName() != group2->getName()) {
      ErrorLog().printf(
            "Groups have different names (\"%s\" versus \"%s\")\n",
            group1->getName().c_str(),
            group2->getName().c_str());
      status = PV_FAILURE;
   }
   if (group1->getKeyword() != group2->getKeyword()) {
      ErrorLog().printf(
            "Groups have different keywords (\"%s\" versus \"%s\").\n",
            group1->getKeyword().c_str(),
            group2->getKeyword().c_str());
      status = PV_FAILURE;
   }

   for (auto const &p : *group1) {
      std::string const &name1 = p.first;
      Parameter const &param1 = p.second;
      bool foundIn2 = group2->present(name1);
      if (!foundIn2) {
         ErrorLog().printf(
               "In group \"%s\", file 1 contains parameter \"%s\" but file 2 does not.\n",
               group1->getName().c_str(), name1.c_str()); 
         status = PV_FAILURE;
         continue;
      }
      auto type1 = param1.getType();
      switch (type1) {
         case Parameter::Type::Numeric:
            {
               double const *value1Ptr = param1.peek<double>();
               assert(value1Ptr);
               double const *value2Ptr = group2->peek<double>(name1);
               assert(value2Ptr); // Above, we checked that the name was present in group2
               if (*value1Ptr != *value2Ptr) {
                  ErrorLog().printf(
                        "Group \"%s\" numeric parameter %s differs "
                        "(%f versus %f, discrepancy %g)\n",
                        group1->getName().c_str(), name1.c_str(),
                        *value1Ptr, *value2Ptr, *value2Ptr - *value1Ptr);
                  status = PV_FAILURE;
               }
            }
            break;
         case Parameter::Type::Array:
            {
               auto const *value1Ptr = param1.peek<std::vector<double>>();
               assert(value1Ptr);
               auto const *value2Ptr = group2->peek<std::vector<double>>(name1);
               assert(value2Ptr); // Above, we checked that the name was present in group2
               if (value1Ptr->size() != value2Ptr->size()) {
                  ErrorLog().printf(
                        "Group \"%s\" array parameter \"%s\" has different sizes "
                        "(%zu versus %zu)\n",
                        group1->getName().c_str(), name1.c_str(),
                        value1Ptr->size(), value2Ptr->size());
                  status = PV_FAILURE;
               }
               else {
                  auto N = value1Ptr->size();
                  for (decltype(N) n = static_cast<decltype(N)>(0); n < N; ++n) {
                     double value1 = (*value1Ptr)[n];
                     double value2 = (*value2Ptr)[n];
                     if (value1 != value2) {
                        ErrorLog().printf(
                              "Group \"%s\" array parameter \"%s\", element %zu differs "
                              "(%f versus %f, discrepancy %g)\n",
                              group1->getName().c_str(), name1.c_str(), n,
                              value1, value2, value2 - value1);
                        status = PV_FAILURE;
                     }
                  }
               }
            }
            break;
         case Parameter::Type::String:
            {
               std::string const *value1Ptr = param1.peek<std::string>();
               assert(value1Ptr);
               auto const *value2Ptr = group2->peek<std::string>(name1);
               assert(value2Ptr); // Above, we checked that the name was present in group2
               if (*value1Ptr != *value2Ptr) {
                  ErrorLog().printf(
                        "Group \"%s\" string parameter %s differs (%s versus %s)\n",
                        group1->getName().c_str(), name1.c_str(),
                        value1Ptr->c_str(), value2Ptr->c_str());
               }
            }
            break;
         default:
            Fatal().printf(
                  "Group 1 parameter \"%s\" has unrecognized type %d\n", name1.c_str(), type1);
            break;
      }
   }

   for (auto const &p : *group2) {
      std::string const &name2 = p.first;
      Parameter const &param2 = p.second;
      bool foundIn1 = group1->present(name2);
      if (!foundIn1) {
         ErrorLog().printf(
               "In group \"%s\", file 2 contains parameter \"%s\" but file 1 does not.\n",
               group2->getName().c_str(), name2.c_str()); 
         status = PV_FAILURE;
         continue;
      }
   }
   return status;
}

} // end namespace PV
