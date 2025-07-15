/*
 * PVParams.hpp
 *
 *  Created on: Nov 27, 2008
 *      Author: rasmussn
 */

#ifndef PVPARAMS_HPP_
#define PVPARAMS_HPP_

#include "arch/mpi/mpi.h"
#include "include/pv_common.h"
#include "params/ParameterSweep.hpp"
#include "params/ParamGroup.hpp"
#include "params/ParamGroupList.hpp"
#include "params/ParamsIO.hpp"
#include "utils/PVLog.hpp"
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace PV {

class PVParams {
  public:
   PVParams(char const *filename, MPI_Comm mpiComm);
   PVParams(char const *buffer, long int bufferLength, MPI_Comm mpiComm);
   virtual ~PVParams();

   int addDefaults(std::string const &path);

   std::shared_ptr<ParamGroup> group(std::string const &groupName);
   std::shared_ptr<ParamGroup const> group(std::string const &groupName) const;
   std::shared_ptr<ParamGroup> defaultGroup(std::string const &keyword);
   std::shared_ptr<ParamGroup const> defaultGroup(std::string const &keyword) const;
   char const *groupNameFromIndex(int index);
   char const *groupKeywordFromIndex(int index);
   char const *groupKeywordFromName(const char *name);

   std::shared_ptr<ParamsIO> makeParamsIO(std::string const &name);
   std::shared_ptr<ParamsIO> makeParamsIO(std::string const &name, std::string const &keyword);

   /**
    * lookForUnread() tests each parameter in each parameter group for whether it's been read.
    * The return value is a vector of pairs of strings. Each pair consists of a group name
    * and a parameter name, indicating that that parameter within that group has not been read.
    */
   std::vector<std::pair<std::string, std::string>> lookForUnread();
   bool hasBeenRead(const char *group_name, const char *param_name);

   int setParameterSweepValues(int n) { return mGroups.setParameterSweepValues(n); }

   /**
    * Randomly shuffles the vector of pointers to the ParamGroup objects.
    * Used for debugging purposes, to help identify cases where behavior depends
    * on the order of objects in the params file during debugging.
    * The shuffling here has no effect on the RNGs managed by the HyPerCol and used
    * by layers or connections.
    */
   void shuffleGroups(unsigned int seed);

   int getNumGroups() { return static_cast<int>(mGroups.size()); }
   int getParameterSweepSize() { return mGroups.getParameterSweepSize(); }

  private:
   void addDefaultParams();
   void addGroup(char *keyword, char *name);
   bool hasSweepValue(const char *paramName) { return mGroups.hasSweepValue(paramName); }
   int clearHasBeenReadFlags();

  private:
   ParamGroupList mGroups;
   ParamGroup mInitialGroup = ParamGroup("", "", 0);
   MPI_Comm mMPIComm;
   int mWorldRank;

   ParamGroupList mDefaultParams;
};

} // end namespace PV

#endif /* PVPARAMS_HPP_ */
