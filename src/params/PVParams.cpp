/*
 * PVParams.cpp
 *
 *  Created on: Nov 27, 2008
 *      Author: rasmussn
 */

#include "PVParams.hpp"
#include "include/pv_common.h"
#include <algorithm> // shuffle, used in shuffleGroups(); transform, used to convert to lower case
#include <cassert>
#include <climits> // INT_MIN
#include <cmath> // nearbyint()
#include <cstdio>
#include <cstdlib>
#include <cstring> // strcmp(), strcpy()
#include <iostream>
#include <random> // mt19937, used in shuffleGroups()

namespace PV {

PVParams::PVParams(char const *filename, MPI_Comm mpiComm) {
   mMPIComm = mpiComm;
   MPI_Comm_rank(mMPIComm, &mWorldRank);
   int parseStatus = mGroups.parseFile(filename, mpiComm);
   FatalIf(parseStatus != PV_SUCCESS, "Failed to parse params file \"%s\"\n", filename);
   addDefaultParams();
}

PVParams::PVParams(char const *buffer, long int bufferLength, MPI_Comm mpiComm) {
   mMPIComm = mpiComm;
   MPI_Comm_rank(mMPIComm, &mWorldRank);
   int parseStatus = mGroups.parseBuffer(buffer, bufferLength);
   FatalIf(parseStatus != PV_SUCCESS, "Failed to parse params\n");
   addDefaultParams();
}

PVParams::~PVParams() {}

void PVParams::addDefaultParams() {
   std::string const &defaultParams = mGroups.getDefaultParamsPath();
   if (defaultParams.empty()) { return; }
   InfoLog() << "Reading default params from " << defaultParams << "\n";
   int parseStatus = addDefaults(defaultParams.c_str());
   FatalIf(
         parseStatus != PV_SUCCESS,
         "Failed to parse default params file \"%s\"\n",
         defaultParams.c_str());
}

int PVParams::addDefaults(std::string const &path) {
   return mDefaultParams.parseFile(path.c_str(), mMPIComm);
}

std::shared_ptr<ParamsIO> PVParams::makeParamsIO(
      std::string const &name) {
   auto paramGroup = group(name);
   return makeParamsIO(name, group(name)->getKeyword());
}

std::shared_ptr<ParamsIO> PVParams::makeParamsIO(
      std::string const &name, std::string const &keyword) {
   return std::make_shared<ParamsIO>(group(name), defaultGroup(keyword));
}

std::shared_ptr<ParamGroup> PVParams::group(std::string const &groupName) {
   return mGroups.group(groupName);
}

std::shared_ptr<ParamGroup const> PVParams::group(std::string const &groupName) const {
   return mGroups.group(groupName);
}

std::shared_ptr<ParamGroup> PVParams::defaultGroup(std::string const &keyword) {
   return mDefaultParams.group(keyword);
}

std::shared_ptr<ParamGroup const> PVParams::defaultGroup(std::string const &keyword) const {
   return mDefaultParams.group(keyword);
}

char const *PVParams::groupNameFromIndex(int index) {
   bool inbounds = index >= 0 && index < getNumGroups();
   return inbounds ? mGroups[index]->getName().c_str() : nullptr;
}

char const *PVParams::groupKeywordFromIndex(int index) {
   bool inbounds = index >= 0 && index < getNumGroups();
   return inbounds ? mGroups[index]->getKeyword().c_str() : nullptr;
}

char const *PVParams::groupKeywordFromName(const char *name) {
   const char *kw    = nullptr;
   std::shared_ptr<ParamGroup> g = group(name);
   if (g != nullptr) {
      kw = g->getKeyword().c_str();
   }
   return kw;
}

void PVParams::addGroup(char *keyword, char *name) {
   mGroups.addGroup(std::string(keyword), std::string(name));
}

std::vector<std::pair<std::string, std::string>> PVParams::lookForUnread() {
   std::vector<std::pair<std::string, std::string>> result(0);
   for (int i = 0; i < getNumGroups(); i++) {
      std::string const &name = mGroups[i]->getName();
      auto unreadParams = mGroups[i]->lookForUnread();
      for (auto const &u : unreadParams) {
         result.emplace_back(mGroups[i]->getName(), u);
      }
   }
   return result;
}

bool PVParams::hasBeenRead(const char *group_name, const char *param_name) {
   std::shared_ptr<ParamGroup> g = group(group_name);
   if (g == nullptr) {
      return false;
   }

   return g->hasBeenRead(param_name);
}

int PVParams::clearHasBeenReadFlags() {
   for (auto &g : mGroups) {
      g->clearAllHasBeenReadFlags();
   }
   return PV_SUCCESS;
}

void PVParams::shuffleGroups(unsigned int seed) {
   if (seed and getNumGroups() > 1) {
      std::mt19937 shuffleRNG(seed);
      std::shuffle(mGroups.begin() + 1, mGroups.end(), shuffleRNG);
   }
}

} // namespace PV
