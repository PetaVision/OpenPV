#ifndef PARAMGROUPLIST_HPP_
#define PARAMGROUPLIST_HPP_

#include "cMakeHeader.h"

#include "arch/mpi/mpi.h"
#include "params/ParamGroup.hpp"
#include "params/ParameterSweep.hpp"

#include <memory>
#include <string>
#include <vector>

namespace PV {

class ParamGroupList {
  public:
   typedef std::vector<std::shared_ptr<ParamGroup>>::value_type value_type;
   typedef std::vector<std::shared_ptr<ParamGroup>>::reference reference;
   typedef std::vector<std::shared_ptr<ParamGroup>>::const_reference const_reference;
   typedef std::vector<std::shared_ptr<ParamGroup>>::iterator iterator;
   typedef std::vector<std::shared_ptr<ParamGroup>>::const_iterator const_iterator;
   typedef std::vector<std::shared_ptr<ParamGroup>>::difference_type difference_type;
   typedef std::vector<std::shared_ptr<ParamGroup>>::size_type size_type;

   ParamGroupList(int processRank);
   ParamGroupList();
   ~ParamGroupList();

   /**
    * Adds a new parameter group with the given keyword and name to the group list
    */
   void addGroup(std::string const &keyword, std::string const &name);

   /**
    * Returns a pointer to the ParameterGroup whose name is given.
    * If there is no such ParameterGroup, returns nullptr
    */
   std::shared_ptr<ParamGroup> group(std::string const &groupName);
   std::shared_ptr<ParamGroup const> group(std::string const &groupName) const;
   bool hasSweepValue(const char *paramName);

   int parseBuffer(char const *buffer, long int bufferLength);
   int parseFile(const char *filename, MPI_Comm mpiComm);
   int setParameterSweepValues(int n);

   std::string const &getDefaultParamsPath() const { return mDefaultParamsPath; }
   int getNumParamSweeps() const { return static_cast<int>(mParamSweeps.size()); }
   int getParameterSweepSize() { return mParameterSweepSize; }

   void action_pvparams_directive(char *id, double val);
   void action_pvparams_filename_directive(char *id, char *stringval);
   void action_parameter_group_name(char *keyword, char *name);
   void action_parameter_group();
   void action_parameter_numeric_def(char *id, double val);
   void action_parameter_numeric_def_overwrite(char *id, double val);
   void action_parameter_array(char *id);
   void action_parameter_array_overwrite(char *id);
   void action_parameter_array_value(double val);
   void action_parameter_string_def(const char *id, const char *stringval);
   void action_parameter_string_def_overwrite(const char *id, const char *stringval);
   void action_parameter_filename_def(const char *id, const char *stringval);
   void action_parameter_filename_def_overwrite(const char *id, const char *stringval);
   void action_parameter_remove(char *id);
   void action_include_directive(const char *stringval);

   void action_parameter_sweep_open(const char *groupname, const char *paramname);
   void action_parameter_sweep_close();
   void action_parameter_sweep_values_number(double val);
   void action_parameter_sweep_values_string(const char *stringval);
   void action_parameter_sweep_values_filename(const char *stringval);

   iterator begin() { return mGroupList.begin(); }
   const_iterator begin() const { return mGroupList.begin(); }
   const_iterator cbegin() const { return mGroupList.begin(); }
   iterator end() { return mGroupList.end(); }
   const_iterator end() const { return mGroupList.end(); }
   const_iterator cend() const { return mGroupList.end(); }
   bool operator==(ParamGroup const &rhs) const;
   bool operator!=(ParamGroup const &rhs) const { return !(*this == rhs); }
   void swap(ParamGroup &rhs);
   size_type size() { return mGroupList.size(); }
   size_type max_size() const { return mGroupList.max_size(); }
   bool empty() const { return mGroupList.empty(); }

   reference operator[](size_type pos) { return mGroupList[pos]; }
   const_reference operator[](size_type pos) const { return mGroupList[pos]; }

  private:
   void addActiveParamSweep(const char *group_name, const char *param_name);
   void checkDuplicates(const char *paramName);
   void initialize();
   void loadParamBuffer(char const *filename, std::string &paramsFileString);
   int setParameterSweepSize();

   /**
    * If a string has quotes as its first and last character, return the
    * part of the string inside the quotes, e.g. {'"', 'c', 'a', 't', '"'}
    * becomes {'c', 'a', 't'}.  If the string is null or does not have quotes at the
    * beginning and end, return empty string.
    */
   static std::string stripQuotationMarks(char const *s);

  private:
   std::shared_ptr<ParamGroup> mActiveGroup = nullptr;
   std::vector<double> mActiveParamArray;
   std::string mCurrSweepGroupName;
   std::string mCurrSweepParamName;
   bool mDebugParsing             = true;
   std::string mDefaultParamsPath = PV_SHARE_DIR "/" "DefaultParams.txt";
   bool mDisable                  = false;
   std::vector<std::shared_ptr<ParamGroup>> mGroupList;
   int mProcessRank;

   std::vector<ParameterSweep*> mParamSweeps;
   ParameterSweep *mActiveParamSweep;
   int mParameterSweepSize; // The number of parameter value sets in the sweep.  Each ParameterSweep
   // group in the params file must contain the same number of values, which is ParameterSweepSize.
};

} // namespace PV

#endif // PARAMGROUPLIST_HPP_
