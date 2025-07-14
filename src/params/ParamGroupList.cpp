#include "ParamGroupList.hpp"

#include "include/pv_common.h"
#include "io/fileio.hpp"
#include "utils/PVLog.hpp"

#include <algorithm> // shuffle, used in shuffleGroups(); transform, used to convert to lower case
#include <cstring>   // strerror, used in parseFile()
#include <fstream>

#ifdef PV_USE_LUA
#include <lua.hpp>
#endif // PV_USE_LUA

// define for debug output
#define DEBUG_PARSING

int pv_parseParameters(PV::ParamGroupList *action_handler, const char *paramBuffer, size_t len);
// pv_parseParameters() is defined in params.y

namespace PV {

ParamGroupList::ParamGroupList() : mProcessRank(0) {
   initialize();
}

ParamGroupList::ParamGroupList(int processRank) : mProcessRank(processRank) {
   initialize();
}

ParamGroupList::~ParamGroupList() {
   for (auto *p : mParamSweeps) {
      delete p;
      p = nullptr;
   }
   delete mActiveParamSweep;
   mActiveParamSweep = nullptr;
}

void ParamGroupList::initialize() {
#ifdef DEBUG_PARSING
   mDebugParsing = true;
#else
   mDebugParsing = false;
#endif // DEBUG_PARSING
   mParamSweeps.clear();
   mActiveParamSweep = new ParameterSweep();
}

void ParamGroupList::action_pvparams_directive(char *id, double val) {
   if (std::string("debugParsing") == id) {
      mDebugParsing = (val != 0);
      if (mProcessRank == 0) {
         InfoLog() << "debugParsing turned " << (mDebugParsing ? "on" : "off") << ".\n";
      }
   }
   else if (std::string("disable") == id) {
      mDisable = (val != 0);
      if (mProcessRank == 0) {
         InfoLog() << "Parsing params file " << (mDisable ? "disabled" : "enabled") << ".\n";
      }
   }
   else {
      if (mProcessRank == 0) {
         WarnLog().printf("Unrecognized directive %s = %f; skipping.\n", id, val);
      }
   }
}

void ParamGroupList::action_pvparams_filename_directive(char *id, char *stringval) {
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_pvparams_filename_directive: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   std::string directive_value = stripQuotationMarks(stringval);
   std::string directive_id(id);
   std::transform(directive_id.begin(), directive_id.end(), directive_id.begin(), ::tolower);
   if (directive_id == "defaultparams") {
      if (!directive_value.empty()) {
         InfoLog().printf("Default params set to \"%s\"\n", directive_value.c_str());
         mDefaultParamsPath = directive_value.c_str();
      }
      else {
         WarnLog().printf("Default params set to NULL; ignored.\n");
      }
   }
}

void ParamGroupList::action_parameter_group() {
   if (mDisable) { return; }
   FatalIf(
         mActiveGroup == nullptr,
         "action_parameter_group() called without an active group defined.\n");
   if (mActiveGroup->empty()) {
      if (mProcessRank == 0) {
         WarnLog().printf(
               "action_parameter_group() called on empty group %s \"%s\"\n",
               mActiveGroup->getKeyword().c_str(),
               mActiveGroup->getName().c_str());
      }
   }
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().printf(
            "action_parameter_group: %s \"%s\" parsed successfully.\n",
            mActiveGroup->getKeyword().c_str(),
            mActiveGroup->getName().c_str());
      InfoLog().flush();
   }
}

void ParamGroupList::action_parameter_group_name(char *keyword, char *name) {
   if (mDisable) { return; }
   FatalIf(
         keyword == nullptr or name == nullptr,
         "action_parameter_group_name() called with null keyword or name");
   std::string nameNoQuotes = stripQuotationMarks(name);
   if (nameNoQuotes.empty()) {
      nameNoQuotes = name;
   }
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().printf(
            "action_parameter_group_name: %s \"%s\" parsed successfully.\n",
            keyword, nameNoQuotes.c_str());
      InfoLog().flush();
   }
   addGroup(keyword, nameNoQuotes.c_str());
}

void ParamGroupList::action_parameter_numeric_def(char *id, double val) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_def: %s = %lf\n", id, val);
      InfoLog().flush();
   }
   checkDuplicates(id);
   mActiveGroup->insert<double>(id, val);
}

void ParamGroupList::action_parameter_numeric_def_overwrite(char *id, double val) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_def_overwrite: %s = %lf\n", id, val);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   FatalIf(
         mActiveGroup == nullptr,
         "action_parameter_numeric_def_overwrite called without an active group defined.\n");
   bool replaceSucceeded = mActiveGroup->replace<double>(id, val);
   FatalIf(
         !replaceSucceeded,
         "Overwrite: %s is not an existing parameter in parameter group %s.\n",
         id, mActiveGroup->getName().c_str());
}

void ParamGroupList::action_parameter_array(char *id) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_array: %s\n", id);
      InfoLog().flush();
   }
   checkDuplicates(id);
   mActiveGroup->insert(id, mActiveParamArray);
   mActiveParamArray.clear();
}

void ParamGroupList::action_parameter_array_overwrite(char *id) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_array_overwrite: %s\n", id);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   bool replaceSucceeded = mActiveGroup->replace<std::vector<double>>(id, mActiveParamArray);
   FatalIf(
         !replaceSucceeded,
         "Overwrite: %s is not an existing parameter in parameter group %s.\n",
         id, mActiveGroup->getName().c_str());
   mActiveParamArray.clear();
}

void ParamGroupList::action_parameter_array_value(double val) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_array_value %lf\n", val);
   }
   mActiveParamArray.emplace_back(val);
}

void ParamGroupList::action_parameter_string_def(const char *id, const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_string_def: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   checkDuplicates(id);
   std::string string_value = stripQuotationMarks(stringval);
   // If the length of stringval is at least 3, string_value should be non-empty
   // (first and last characters of stringval are quotation marks; the rest is the
   // value of the string parameter.
   FatalIf(
         stringval != nullptr and std::strlen(stringval) >= 3 and string_value.empty(),
         "action_parameter_string_def() received a bad string value: %s\n"
         "If non-empty, the string value should be enclosed in quotation marks.\n",
         stringval);
   mActiveGroup->insert<std::string>(id, string_value);
}

void ParamGroupList::action_parameter_string_def_overwrite(const char *id, const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_string_def_overwrite: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   bool replaceSucceeded = mActiveGroup->replace<std::string>(id, stringval);
   FatalIf(
         !replaceSucceeded,
         "Overwrite: %s is not an existing parameter in parameter group %s.\n",
         id, mActiveGroup->getName().c_str());
}

void ParamGroupList::action_parameter_filename_def(const char *id, const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_filename_def: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   checkDuplicates(id);
   std::string param_value = stripQuotationMarks(stringval);
   assert(!param_value.empty());
   mActiveGroup->insert<std::string>(id, param_value);
}

void ParamGroupList::action_parameter_filename_def_overwrite(const char *id, const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_filename_def_overwrite: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   std::string param_value = stripQuotationMarks(stringval);
   bool replaceSucceeded   = mActiveGroup->replace<std::string>(id, param_value);
   FatalIf(
         !replaceSucceeded,
         "Overwrite: %s is not an existing parameter in parameter group %s.\n",
         id, mActiveGroup->getName().c_str());
}

void ParamGroupList::action_parameter_remove(char *id) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_def_remove: %s\n", id);
      InfoLog().flush();
   }
   bool eraseSucceeded = mActiveGroup->erase(id);
   FatalIf(
         !eraseSucceeded,
         "Remove: %s is not an existing parameter in parameter group %s.\n",
         id, mActiveGroup->getName().c_str());
}

void ParamGroupList::action_include_directive(const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_include_directive: including %s\n", stringval);
      InfoLog().flush();
   }
   // Grab the included group name
   std::string include_name = stripQuotationMarks(stringval);
   FatalIf(include_name.empty(), "action_include_directive called with bad argument.\n");
   // Grab the included group's ParamGroup object
   std::shared_ptr<ParamGroup> includeGroup = group(std::string(include_name.c_str()));
   // Fail if target group not found
   FatalIf(
         includeGroup == nullptr,
         "Include: include group %s is not defined.\n", include_name.c_str());
   // Check keyword of group
   FatalIf(
         mActiveGroup->getKeyword() != includeGroup->getKeyword(),
         "Include: Cannot include group %s \"%s\" into %s \"%s\". Group types must be the same.\n",
            includeGroup->getKeyword().c_str(),
            include_name.c_str(),
            mActiveGroup->getKeyword().c_str(),
            mActiveGroup->getName().c_str());
   // Load all parameters from include group into current parameter group
   for (auto &p : *includeGroup) {
      switch (p.second.getType()) {
         case Parameter::Type::Numeric:
            {
               auto peekResult = p.second.peek<double>();
               assert(peekResult);
               mActiveGroup->insert<double>(p.first, *peekResult);
            }
            break;
         case Parameter::Type::Array:
            {
               auto peekResult = p.second.peek<std::vector<double>>();
               assert(peekResult);
               mActiveGroup->insert<std::vector<double>>(p.first, *peekResult);
            }
            break;
         case Parameter::Type::String:
            {
               auto peekResult = p.second.peek<std::string>();
               assert(peekResult);
               mActiveGroup->insert<std::string>(p.first, *peekResult);
            }
            break;
         default:
            Fatal().printf(
                  "Parameter group %s \"%s\" parameter %s has unrecognized parameter type %d\n",
                  includeGroup->getKeyword().c_str(),
                  includeGroup->getName().c_str(),
                  p.first.c_str(),
                  p.second.getType());
            break;
      }
   }
}

void ParamGroupList::action_parameter_sweep_open(const char *groupname, const char *paramname) {
   if (mDisable)
      return;
   // strip quotation marks from groupname
   mCurrSweepGroupName = stripQuotationMarks(groupname);
   FatalIf(
         mCurrSweepGroupName == "",
         "action_parameter_sweep_open called without groupname in quotation marks.\n");
   mCurrSweepParamName = paramname;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf(
            "action_parameter_sweep_open: Sweep for group %s, parameter \"%s\" starting\n",
            groupname,
            paramname);
      InfoLog().flush();
   }
}

void ParamGroupList::action_parameter_sweep_close() {
   if (mDisable)
      return;
   addActiveParamSweep(mCurrSweepGroupName.c_str(), mCurrSweepParamName.c_str());
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().printf(
            "action_parameter_group: ParameterSweep for %s \"%s\" parsed successfully.\n",
            mCurrSweepGroupName.c_str(),
            mCurrSweepParamName.c_str());
      InfoLog().flush();
   }
   // build a parameter group
   mCurrSweepGroupName = "";
   mCurrSweepParamName = "";
}

void ParamGroupList::action_parameter_sweep_values_number(double val) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_sweep_values_number: %f\n", val);
      InfoLog().flush();
   }
   mActiveParamSweep->pushNumericValue(val);
}

void ParamGroupList::action_parameter_sweep_values_string(const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_sweep_values_string: %s\n", stringval);
      InfoLog().flush();
   }
   std::string string = stripQuotationMarks(stringval);
   // stringval can be null, but if stringval is not null, string should also be non-null
   assert(stringval == nullptr || !string.empty());
   mActiveParamSweep->pushStringValue(string);
}

void ParamGroupList::action_parameter_sweep_values_filename(const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mProcessRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_sweep_values_filename: %s\n", stringval);
      InfoLog().flush();
   }
   std::string filename = stripQuotationMarks(stringval);
   mActiveParamSweep->pushStringValue(filename);
}

void ParamGroupList::addActiveParamSweep(const char *group_name, const char *param_name) {
   // Search for group_name and param_name in both ParameterSweep and BatchSweep list of objects
   for (auto *p : mParamSweeps) {
       assert(p != nullptr);
       FatalIf(
             p->getGroupName() == group_name and p->getParamName() == param_name,
            "ParamGroupList::addActiveParamSweep: Parameter sweep %s, %s already exists\n",
            group_name,
            param_name);
   }

   mActiveParamSweep->setGroupAndParameter(group_name, param_name);
   mParamSweeps.push_back(mActiveParamSweep);
   mActiveParamSweep = new ParameterSweep();
}

void ParamGroupList::addGroup(std::string const &keyword, std::string const &name) {
   // Verify that the new group's name is not an existing group's name
   for (auto const &g : mGroupList) {
      if (g->getName() == name) {
         Fatal().printf(
               "Rank %d process: group name \"%s\" duplicated\n", mProcessRank, name.c_str());
      }
   }

   mGroupList.push_back(std::make_shared<ParamGroup>(name, keyword, mProcessRank));
   mActiveGroup = mGroupList.back();
}

void ParamGroupList::checkDuplicates(const char *paramName) {
   bool hasDuplicate = false;
   bool isPresent = mActiveGroup->present(paramName);
   FatalIf(
         isPresent,
         "Rank %d process: The params group for %s \"%s\") duplicates parameter \"%s\".\n",
         mProcessRank,
         mActiveGroup->getKeyword().c_str(),
         mActiveGroup->getName().c_str(),
         paramName);
}

std::shared_ptr<ParamGroup> ParamGroupList::group(std::string const &groupName) {
   for (auto const &g : mGroupList) {
      if (g->getName() == groupName) {
         return g;
      }
   }
   return nullptr;
}

std::shared_ptr<ParamGroup const> ParamGroupList::group(std::string const &groupName) const {
   for (auto const &g : mGroupList) {
      if (g->getName() == groupName) {
         return g;
      }
   }
   return nullptr;
}

bool ParamGroupList::hasSweepValue(const char *inParamName) {
   bool out = false;
   const char *group_name;
   for (int k = 0; k < getNumParamSweeps(); k++) {
      ParameterSweep *sweep          = mParamSweeps[k];
      std::string const &group_name  = sweep->getGroupName();
      std::string const &param_name  = sweep->getParamName();
      std::shared_ptr<ParamGroup> gp = group(group_name);
      if (gp == nullptr) {
         Fatal().printf(
               "ParamGroupList::hasSweepValue error: ParameterSweep %d (zero-indexed) refers to "
               "non-existent group \"%s\"\n",
               k,
               group_name.c_str());
      }
      if (gp->getKeyword() == "HyPerCol" && param_name != inParamName) {
         out = true;
         break;
      }
   }
   return out;
}

void ParamGroupList::loadParamBuffer(char const *filename, std::string &paramsFileString) {
   if (filename == nullptr) {
      Fatal() << "ParamGroupList::loadParamBuffer: filename is null\n";
   }
   struct stat filestatus;
   if (PV_stat(filename, &filestatus)) {
      Fatal().printf(
            "ParamGroupList::parseFile unable to get status of file \"%s\": %s\n",
            filename,
            std::strerror(errno));
   }
   if (filestatus.st_mode & S_IFDIR) {
      Fatal().printf("ParamGroupList::parseFile: specified file \"%s\" is a directory.\n", filename);
   }

#ifdef PV_USE_LUA
   char const *const luaext = ".lua";
   size_t const luaextlen   = strlen(luaext);
   size_t const fnlen       = strlen(filename);

   bool const useLua = fnlen >= luaextlen && !std::strcmp(&filename[fnlen - luaextlen], luaext);
#else // PV_USE_LUA
   bool const useLua = false;
#endif // PV_USE_LUA

   if (useLua) {
#ifdef PV_USE_LUA
      InfoLog() << "Running lua program \"" << filename << "\".\n";
      lua_State *lua_state = luaL_newstate();
      luaL_openlibs(lua_state);
      int result = luaL_dofile(lua_state, filename);
      if (result != LUA_OK) {
         char const *errorMessage = lua_tostring(lua_state, -1);
         lua_pop(lua_state, 1);
         Fatal() << errorMessage << "\n";
      }
      lua_getglobal(lua_state, "paramsFileString");
      size_t llength;
      char const *lstring = lua_tolstring(lua_state, -1, &llength);
      if (lstring == nullptr) {
         Fatal() << "Lua program \"" << filename
                 << "\" does not create a string variable \"paramsFileString\".\n";
      }
      paramsFileString.insert(paramsFileString.end(), lstring, &lstring[llength]);
      lua_pop(lua_state, 1);
      lua_close(lua_state);
      InfoLog() << "Retrieved paramsFileString, with length " << llength << ".\n";
#endif // PV_USE_LUA
   }
   else {
      off_t sz = filestatus.st_size;
      std::ifstream paramsStream(filename, std::ios_base::in);
      if (paramsStream.fail()) {
         throw;
      } // TODO: provide a helpful strerror(errno)-like message
      paramsFileString.resize(sz);
      paramsStream.read(&paramsFileString[0], sz);
   }
}

int ParamGroupList::parseBuffer(char const *buffer, long int bufferLength) {
   // Assumes that each MPI process has the same contents in buffer.

   // This is where it calls the scanner and parser
   int status = pv_parseParameters(this, buffer, bufferLength);
   if (status != 0) {
      ErrorLog().printf(
            "Rank %d process: pv_parseParameters failed with return value %d\n",
            mProcessRank,
            status);
      return PV_FAILURE;
   }
   getOutputStream().flush();

   // Need to set sweepSize here, because if the outputPath sweep needs to be created
   // we need to know the size.
   setParameterSweepSize();

   // If there is at least one ParameterSweep  and none of them set outputPath, create a
   // parameterSweep that does set outputPath.

   // If both parameterSweep and batchSweep is set, must autoset output path, as there is no way to
   // specify both paramSweep and batchSweep
   if (getNumParamSweeps() > 0) {
      if (!hasSweepValue("outputPath")) {
         const char *hypercolgroupname = nullptr;
         const char *outputPathName    = nullptr;
         for (auto &g : mGroupList) {
            if (g->getKeyword() == "HyPerCol") {
               hypercolgroupname  = g->getName().c_str();
               std::string const *readOutputPath = g->read<std::string>("outputPath");
               if (readOutputPath) {
                  outputPathName = readOutputPath->c_str();
               }
               if (outputPathName[0] == '\0') {
                  Fatal().printf(
                        "HyPerCol parameter outputPath must be specified if parameterSweep does "
                        "not sweep over outputPath\n");
               }
               break;
            }
         }
         if (hypercolgroupname == nullptr) {
            ErrorLog().printf("ParamGroupList::parseBuffer: no HyPerCol group\n");
            abort();
         }

         // Push the strings "[outputPathName]/paramsweep_[n]/"
         // to the parameter sweep, where [n] ranges from 0 to mParameterSweepSize - 1,
         // and is zero-padded so that the parameter sweep's outputPath directories
         // sort the same lexicographically and numerically.
         auto lenmax = std::to_string(mParameterSweepSize - 1).size();
         for (int i = 0; i < mParameterSweepSize; i++) {
            std::string outputPathStr(outputPathName);
            outputPathStr.append("/paramsweep_");
            std::string serialNumberStr = std::to_string(i);
            auto len                    = serialNumberStr.size();
            if (len < lenmax) {
               outputPathStr.append(lenmax - len, '0');
            }
            outputPathStr.append(serialNumberStr);
            outputPathStr.append("/");
            mActiveParamSweep->pushStringValue(outputPathStr);
         }
         addActiveParamSweep(hypercolgroupname, "outputPath");
      }

      if (!hasSweepValue("checkpointWriteDir")) {
         const char *hypercolgroupname      = nullptr;
         std::string const *checkpointWriteDir = nullptr;
         for (auto &g : mGroupList) {
            if (g->getKeyword() == "HyPerCol") {
               hypercolgroupname  = g->getName().c_str();
               checkpointWriteDir = g->read<std::string>("checkpointWriteDir");
               // checkpointWriteDir can be nullptr if checkpointWrite is set to false
               break;
            }
         }
         if (hypercolgroupname == nullptr) {
            ErrorLog().printf("ParamGroupList::parseBuffer: no HyPerCol group\n");
            abort();
         }
         if (checkpointWriteDir) {
            // Push the strings "[checkpointWriteDir]/paramsweep_[n]/"
            // to the parameter sweep, where [n] ranges from 0 to mParameterSweepSize - 1,
            // and is zero-padded so that the parameter sweep's checkpointWriteDir directories
            // sort the same lexicographically and numerically.
            auto lenmax = std::to_string(mParameterSweepSize - 1).size();
            for (int i = 0; i < mParameterSweepSize; i++) {
               std::string checkpointWriteDirStr(*checkpointWriteDir);
               checkpointWriteDirStr.append("/paramsweep_");
               std::string serialNumberStr = std::to_string(i);
               auto len                    = serialNumberStr.size();
               if (len < lenmax) {
                  checkpointWriteDirStr.append(lenmax - len, '0');
               }
               checkpointWriteDirStr.append(serialNumberStr);
               checkpointWriteDirStr.append("/");
               mActiveParamSweep->pushStringValue(checkpointWriteDirStr);
            }
            addActiveParamSweep(hypercolgroupname, "checkpointWriteDir");
         }
      }
   }

   // Each ParameterSweep needs to have its group/parameter pair added to the database, if it's not
   // already present.
   for (int k = 0; k < getNumParamSweeps(); k++) {
      ParameterSweep *sweep  = mParamSweeps[k];
      std::string const &group_name = sweep->getGroupName();
      std::string const &param_name = sweep->getParamName();
      SweepType type                = sweep->getType();
      std::shared_ptr<ParamGroup> g = group(group_name);
      if (g == nullptr) {
         ErrorLog().printf("ParameterSweep: there is no group \"%s\"\n", group_name.c_str());
         abort();
      }
      switch (type) {
         case SWEEP_NUMBER:
            if (!g->present(param_name)) {
               g->insert(param_name, 0.0);
            }
            break;
         case SWEEP_STRING:
            if (!g->present(param_name)) {
               g->insert(param_name, "");
            }
            break;
         default: assert(0); break;
      }
   }

   for (auto &g : mGroupList) {
      g->clearAllHasBeenReadFlags();
   }

   return status;
}

int ParamGroupList::parseFile(const char *filename, MPI_Comm mpiComm) {
   int rootproc      = 0;
   std::string paramBuffer("");
   size_t bufferlen;
   int sz;
   MPI_Comm_size(mpiComm, &sz);
   int rank;
   MPI_Comm_rank(mpiComm, &rank);
   if (rank == rootproc) {
      loadParamBuffer(filename, paramBuffer);
      bufferlen = paramBuffer.size();
      // Older versions of MPI_Send require void*, not void const*

#ifdef PV_USE_MPI
      for (int i = 0; i < sz; i++) {
         if (i == rootproc)
            continue;
         MPI_Send(&paramBuffer[0], (int)bufferlen, MPI_CHAR, i, 31, mpiComm);
      }
#endif // PV_USE_MPI
   }
   else { // rank != rootproc
#ifdef PV_USE_MPI
      MPI_Status mpi_status;
      int count;
      MPI_Probe(rootproc, 31, mpiComm, &mpi_status);
      MPI_Get_count(&mpi_status, MPI_CHAR, &count);
      bufferlen   = (size_t)count;
      paramBuffer.resize(bufferlen);
      MPI_Recv(
            &paramBuffer[0],
            count,
            MPI_CHAR,
            rootproc,
            31,
            mpiComm,
            MPI_STATUS_IGNORE);
#endif // PV_USE_MPI
   }

   int status = parseBuffer(paramBuffer.data(), bufferlen);
   return status;
}

int ParamGroupList::setParameterSweepSize() {
   mParameterSweepSize = -1;
   for (int k = 0; k < getNumParamSweeps(); k++) {
      if (mParameterSweepSize < 0) {
         mParameterSweepSize = mParamSweeps[k]->getNumValues();
         assert(mParameterSweepSize > 0);
      }
      else {
         if (mParameterSweepSize != mParamSweeps[k]->getNumValues()) {
            ErrorLog().printf(
                  "ParamGroupList::setParameterSweepSize: all ParameterSweeps in the "
                  "parameters file must have the same number of entries.\n");
            abort();
         }
      }
   }
   if (mParameterSweepSize < 0)
      mParameterSweepSize = 0;
   return mParameterSweepSize;
}

int ParamGroupList::setParameterSweepValues(int n) {
   int status = PV_SUCCESS;
   // Set parameter sweeps
   if (n < 0 || n >= mParameterSweepSize) {
      status = PV_FAILURE;
      return status;
   }
   for (int k = 0; k < this->getNumParamSweeps(); k++) {
      ParameterSweep *paramSweep     = mParamSweeps[k];
      SweepType type                 = paramSweep->getType();
      std::string const &group_name  = paramSweep->getGroupName();
      std::string const &param_name  = paramSweep->getParamName();
      std::shared_ptr<ParamGroup> gp = group(group_name);
      assert(gp != nullptr);

      const char *s;
      double v = 0.0f;
      switch (type) {
         case SWEEP_NUMBER:
            status = paramSweep->getNumericValue(n, &v);
            gp->replace(param_name, v);
            break;
         case SWEEP_STRING:
            status = paramSweep->getStringValue(n, &s);
            gp->replace(param_name, s);
            break;
         default:
            Fatal().printf(
                  "Unrecognized parameter sweep type for %s \"%s\"\n",
                  group_name.c_str(), param_name.c_str());
            break;
      }
   }
   return status;
}

std::string ParamGroupList::stripQuotationMarks(char const *s) {
   if (s == nullptr) { return ""; }

   std::string quotedString(s);
   int len = static_cast<int>(quotedString.size());
   if (len >= 2 and quotedString[0] == '"' and quotedString[len-1] == '"') {
       std::string noQuotesString = quotedString.substr(1, len - 2);
       return noQuotesString;
   }
   else {
      return "";
   }
}

} // namespace PV
