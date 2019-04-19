/*
 * PVParams.hpp
 *
 *  Created on: Nov 27, 2008
 *      Author: rasmussn
 */

#ifndef PVPARAMS_HPP_
#define PVPARAMS_HPP_

#include "FileStream.hpp"
#include "columns/Communicator.hpp"
#include "fileio.hpp"
#include "include/pv_common.h"
#include "io.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>

// TODO - make MAX_PARAMS dynamic
#define MAX_PARAMS 100 // maximum number of parameters in a group

namespace PV {

enum ParamsIOFlag { PARAMS_IO_READ, PARAMS_IO_WRITE };

class Parameter {
  public:
   Parameter(const char *name, double value);
   virtual ~Parameter();

   const char *name() { return paramName; }
   double value() {
      hasBeenReadFlag = true;
      return paramDblValue;
   }
   const float *valuePtr() {
      hasBeenReadFlag = true;
      return &paramValue;
   }
   const double *valueDblPtr() {
      hasBeenReadFlag = true;
      return &paramDblValue;
   }
   bool hasBeenRead() { return hasBeenReadFlag; }
   void clearHasBeenRead() { hasBeenReadFlag = false; }
   void setValue(double v) {
      paramValue    = (float)v;
      paramDblValue = v;
   }
   Parameter *copyParameter() { return new Parameter(paramName, paramDblValue); }

  private:
   char *paramName;
   float paramValue;
   double paramDblValue;
   bool hasBeenReadFlag;
};

class ParameterArray {
  public:
   ParameterArray(int initialSize);
   virtual ~ParameterArray();
   int getArraySize() { return arraySize; }
   const char *name() { return paramName; }
   int setName(const char *name);
   const float *getValues(int *sz) {
      hasBeenReadFlag = true;
      *sz             = arraySize;
      return values;
   }
   const double *getValuesDbl(int *sz) {
      hasBeenReadFlag = true;
      *sz             = arraySize;
      return valuesDbl;
   }
   int pushValue(double value);
   void resetArraySize() { arraySize = 0; }
   bool hasBeenRead() { return hasBeenReadFlag; }
   void clearHasBeenRead() { hasBeenReadFlag = false; }
   double peek(int index) { return valuesDbl[index]; }
   ParameterArray *copyParameterArray();

  private:
   bool paramNameSet;
   char *paramName;
   int arraySize; // The number of values that have been pushed
   int bufferSize; // The size of the buffer in memory
   double *valuesDbl;
   float *values;
   bool hasBeenReadFlag;
};

class ParameterString {
  public:
   ParameterString(const char *name, const char *value);
   virtual ~ParameterString();

   const char *getName() { return paramName; }
   const char *getValue() {
      hasBeenReadFlag = true;
      return paramValue;
   }
   bool hasBeenRead() { return hasBeenReadFlag; }
   void clearHasBeenRead() { hasBeenReadFlag = false; }
   void setValue(const char *s) {
      free(paramValue);
      paramValue = s ? strdup(s) : NULL;
   }
   ParameterString *copyParameterString() { return new ParameterString(paramName, paramValue); }

  private:
   char *paramName;
   char *paramValue;
   bool hasBeenReadFlag;
};

class ParameterStack {
  public:
   ParameterStack(int maxCount);
   virtual ~ParameterStack();

   int push(Parameter *param);
   Parameter *pop();
   Parameter *peek(int index) { return parameters[index]; }
   int size() { return count; }

  private:
   int count;
   int maxCount;
   Parameter **parameters;
};

class ParameterArrayStack {
  public:
   ParameterArrayStack(int initialCount);
   virtual ~ParameterArrayStack();
   int push(ParameterArray *array);
   int size() { return count; }
   ParameterArray *peek(int index) {
      return index >= 0 && index < count ? parameterArrays[index] : NULL;
   }

  private:
   int count; // Number of ParameterArrays
   int allocation; // Size of buffer
   ParameterArray **parameterArrays;
};

class ParameterStringStack {
  public:
   ParameterStringStack(int initialCount);
   virtual ~ParameterStringStack();

   int push(ParameterString *param);
   ParameterString *pop();
   ParameterString *peek(int index) {
      return index >= 0 && index < count ? parameterStrings[index] : NULL;
   }
   int size() { return count; }
   const char *lookup(const char *targetname);

  private:
   int count;
   int allocation;
   ParameterString **parameterStrings;
};

class ParameterGroup {
  public:
   ParameterGroup(
         char *name,
         ParameterStack *stack,
         ParameterArrayStack *array_stack,
         ParameterStringStack *string_stack,
         int rank = 0);
   virtual ~ParameterGroup();

   const char *name() { return groupName; }
   const char *getGroupKeyword() { return groupKeyword; }
   int setGroupKeyword(const char *keyword);
   int setStringStack(ParameterStringStack *stringStack);
   int present(const char *name);
   double value(const char *name);
   bool arrayPresent(const char *name);
   const float *arrayValues(const char *name, int *size);
   const double *arrayValuesDbl(const char *name, int *size);
   int stringPresent(const char *stringName);
   const char *stringValue(const char *stringName);
   int warnUnread();
   bool hasBeenRead(const char *paramName);
   int clearHasBeenReadFlags();
   int pushNumerical(Parameter *param);
   int pushString(ParameterString *param);
   int setValue(const char *param_name, double value);
   int setStringValue(const char *param_name, const char *svalue);
   ParameterStack *copyStack();
   ParameterArrayStack *copyArrayStack();
   ParameterStringStack *copyStringStack();

  private:
   char *groupName;
   char *groupKeyword;
   ParameterStack *stack;
   ParameterArrayStack *arrayStack;
   ParameterStringStack *stringStack;
   int processRank;
};

enum SweepType { SWEEP_UNDEF = 0, SWEEP_NUMBER = 1, SWEEP_STRING = 2 };

class ParameterSweep {
  public:
   ParameterSweep();
   virtual ~ParameterSweep();

   int setGroupAndParameter(const char *groupname, const char *paramname);
   int pushNumericValue(double val);
   int pushStringValue(const char *sval);
   int getNumValues() { return numValues; }
   SweepType getType() { return type; }
   int getNumericValue(int n, double *val);
   const char *getStringValue(int n);
   const char *getGroupName() { return groupName; }
   const char *getParamName() { return paramName; }

  private:
   char *groupName;
   char *paramName;
   SweepType type;
   int numValues;
   int currentBufferSize;
   double *valuesNumber;
   char **valuesString;
};

class PVParams {
  public:
   PVParams(size_t initialSize, Communicator const *inIcComm);
   PVParams(const char *filename, size_t initialSize, Communicator const *inIcComm);
   PVParams(
         const char *buffer,
         long int bufferLength,
         size_t initialSize,
         Communicator const *inIcComm);
   virtual ~PVParams();

   bool getParseStatus() { return parseStatus; }

   template <typename T>
   void ioParamValueRequired(
         enum ParamsIOFlag ioFlag,
         const char *groupName,
         const char *paramName,
         T *val);
   template <typename T>
   void ioParamValue(
         enum ParamsIOFlag ioFlag,
         const char *groupName,
         const char *paramName,
         T *val,
         T defaultValue,
         bool warnIfAbsent = true);
   void ioParamString(
         enum ParamsIOFlag ioFlag,
         const char *groupName,
         const char *paramName,
         char **paramStringValue,
         const char *defaultValue,
         bool warnIfAbsent = true);

   void ioParamStringRequired(
         enum ParamsIOFlag ioFlag,
         const char *groupName,
         const char *paramName,
         char **paramStringValue);
   template <typename T>
   void ioParamArray(
         enum ParamsIOFlag ioFlag,
         const char *groupName,
         const char *paramName,
         T **paramArrayValue,
         int *arraysize);
   template <typename T>
   void writeParam(const char *paramName, T paramValue);
   template <typename T>
   void writeParamArray(const char *paramName, const T *array, int arraysize);
   void writeParamString(const char *paramName, const char *svalue);

   int present(const char *groupName, const char *paramName);
   double value(const char *groupName, const char *paramName);
   double
   value(const char *groupName,
         const char *paramName,
         double initialValue,
         bool warnIfAbsent = true);
   int valueInt(const char *groupName, const char *paramName);
   int valueInt(
         const char *groupName,
         const char *paramName,
         int initialValue,
         bool warnIfAbsent = true);
   bool arrayPresent(const char *groupName, const char *paramName);
   const float *arrayValues(
         const char *groupName,
         const char *paramName,
         int *arraySize,
         bool warnIfAbsent = true);
   const double *arrayValuesDbl(
         const char *groupName,
         const char *paramName,
         int *arraySize,
         bool warnIfAbsent = true);
   int stringPresent(const char *groupName, const char *paramStringName);
   const char *
   stringValue(const char *groupName, const char *paramStringName, bool warnIfAbsent = true);
   ParameterGroup *group(const char *groupName);
   const char *groupNameFromIndex(int index);
   const char *groupKeywordFromIndex(int index);
   const char *groupKeywordFromName(const char *name);
   int warnUnread();
   bool hasBeenRead(const char *group_name, const char *param_name);
   bool presentAndNotBeenRead(const char *group_name, const char *param_name);
   void handleUnnecessaryParameter(const char *group_name, const char *param_name);
   template <typename T>
   void handleUnnecessaryParameter(const char *group_name, const char *param_name, T correct_value);

   /**
    * If the given parameter group has a string parameter with the given parameter name,
    * issue a warning that the string parameter is unnecessary, and mark string parameter as having
    * been read.
    */
   void handleUnnecessaryStringParameter(const char *group_name, const char *param_name);

   /**
    * If the given parameter group has a string parameter with the given parameter name,
    * issue a warning that the string parameter is unnecessary, and mark string parameter as having
    * been read.
    * Additionally, compare the value in params to the given correct value, and exit with an error
    * if they
    * are not equal.
    */
   void handleUnnecessaryStringParameter(
         const char *group_name,
         const char *param_name,
         const char *correctValue,
         bool case_insensitive_flag = false);

   void setPrintLuaStream(FileStream *printLuaStream) { mPrintLuaStream = printLuaStream; }
   void setPrintParamsStream(FileStream *printParamsStream) {
      mPrintParamsStream = printParamsStream;
   }
   int setParameterSweepValues(int n);

   void action_pvparams_directive(char *id, double val);
   void action_parameter_group_name(char *keyword, char *name);
   void action_parameter_group();
   void action_parameter_def(char *id, double val);
   void action_parameter_def_overwrite(char *id, double val);
   void action_parameter_array(char *id);
   void action_parameter_array_overwrite(char *id);
   void action_parameter_array_value(double val);
   void action_parameter_string_def(const char *id, const char *stringval);
   void action_parameter_string_def_overwrite(const char *id, const char *stringval);
   void action_parameter_filename_def(const char *id, const char *stringval);
   void action_parameter_filename_def_overwrite(const char *id, const char *stringval);
   void action_include_directive(const char *stringval);

   void action_parameter_sweep_open(const char *groupname, const char *paramname);
   void action_parameter_sweep_close();
   void action_parameter_sweep_values_number(double val);
   void action_parameter_sweep_values_string(const char *stringval);
   void action_parameter_sweep_values_filename(const char *stringval);

   int numberOfGroups() { return numGroups; }
   int numberOfParameterSweeps() { return numParamSweeps; }
   int getParameterSweepSize() { return parameterSweepSize; }
   FileStream *getPrintParamsStream() { return mPrintParamsStream; }
   FileStream *getPrintLuaStream() { return mPrintLuaStream; }

  private:
   int parseStatus;
   int numGroups;
   size_t groupArraySize;
   ParameterGroup **groups;
   ParameterStack *stack;
   ParameterArrayStack *arrayStack;
   ParameterStringStack *stringStack;
   bool debugParsing;
   bool disable;
   Communicator const *icComm;
   int worldRank;
   int worldSize;

   ParameterArray *currentParamArray;

   int numParamSweeps; // The number of different parameters that are changed during the sweep.
   ParameterSweep **paramSweeps;
   ParameterSweep *activeParamSweep;
   int parameterSweepSize; // The number of parameter value sets in the sweep.  Each ParameterSweep
   // group in the params file must contain the same number of values, which
   // is sweepSize.

   char *currGroupKeyword;
   char *currGroupName;

   char *currSweepGroupName;
   char *currSweepParamName;

   FileStream *mPrintParamsStream = nullptr;
   FileStream *mPrintLuaStream    = nullptr;

   int initialize(size_t initialSize);
   int parseFile(const char *filename);
   void loadParamBuffer(char const *filename, std::string &paramsFileString);
   int parseBuffer(const char *buffer, long int bufferLength);
   int setParameterSweepSize();
   void addGroup(char *keyword, char *name);
   void addActiveParamSweep(const char *group_name, const char *param_name);
   void checkDuplicates(const char *paramName);
   int newActiveParamSweep();
   int clearHasBeenReadFlags();
   static char *stripQuotationMarks(const char *s);
   static char *stripOverwriteTag(const char *s);
   bool hasSweepValue(const char *paramName);
   int convertParamToInt(double value);
};

template <typename T>
void PVParams::handleUnnecessaryParameter(
      const char *group_name,
      const char *param_name,
      T correct_value) {
   int status = PV_SUCCESS;
   if (present(group_name, param_name)) {
      if (worldRank == 0) {
         const char *class_name = groupKeywordFromName(group_name);
         WarnLog().printf(
               "%s \"%s\" does not use parameter %s, but it is present in the parameters file.\n",
               class_name,
               group_name,
               param_name);
      }
      T params_value = (T)value(
            group_name,
            param_name); // marks param as read so that presentAndNotBeenRead doesn't trip up
      if (params_value != correct_value) {
         status = PV_FAILURE;
         if (worldRank == 0) {
            ErrorLog() << "   Value " << params_value << " is inconsistent with correct value "
                       << correct_value << std::endl;
         }
      }
   }
   MPI_Barrier(icComm->globalCommunicator());
   if (status != PV_SUCCESS)
      exit(EXIT_FAILURE);
}

template <typename T>
void PVParams::ioParamValueRequired(
      enum ParamsIOFlag ioFlag,
      const char *groupName,
      const char *paramName,
      T *paramValue) {
   switch (ioFlag) {
      case PARAMS_IO_READ: *paramValue = (T)value(groupName, paramName); break;
      case PARAMS_IO_WRITE: writeParam(paramName, *paramValue); break;
   }
}

template <typename T>
void PVParams::ioParamValue(
      enum ParamsIOFlag ioFlag,
      const char *groupName,
      const char *paramName,
      T *paramValue,
      T defaultValue,
      bool warnIfAbsent) {
   switch (ioFlag) {
      case PARAMS_IO_READ:
         *paramValue = (T)value(groupName, paramName, defaultValue, warnIfAbsent);
         break;
      case PARAMS_IO_WRITE: writeParam(paramName, *paramValue); break;
   }
}
template <typename T>
void PVParams::ioParamArray(
      enum ParamsIOFlag ioFlag,
      const char *groupName,
      const char *paramName,
      T **paramArrayValue,
      int *arraysize) {
   if (ioFlag == PARAMS_IO_READ) {
      const double *paramArray = arrayValuesDbl(groupName, paramName, arraysize);
      pvAssert(*arraysize >= 0);
      if (*arraysize > 0) {
         *paramArrayValue = (T *)calloc((size_t)*arraysize, sizeof(T));
         FatalIf(
               paramArrayValue == nullptr,
               "%s \"%s\": global rank %d process unable to copy array parameter %s: %s\n",
               groupKeywordFromName(groupName),
               groupName,
               icComm->globalCommRank(),
               paramName,
               strerror(errno));
         for (int k = 0; k < *arraysize; k++) {
            (*paramArrayValue)[k] = (T)paramArray[k];
         }
      }
      else {
         *paramArrayValue = nullptr;
      }
   }
   else if (ioFlag == PARAMS_IO_WRITE) {
      writeParamArray(paramName, *paramArrayValue, *arraysize);
   }
}

template <typename T>
void PVParams::writeParam(const char *paramName, T paramValue) {
   if (mPrintParamsStream) {
      pvAssert(mPrintLuaStream);
      std::stringstream vstr("");
      if (std::numeric_limits<T>::has_infinity) {
         if (paramValue == std::numeric_limits<T>::min()) {
            vstr << "-infinity";
         }
         else if (paramValue == std::numeric_limits<T>::max()) {
            vstr << "infinity";
         }
         else {
            vstr << paramValue;
         }
      }
      else {
         vstr << paramValue;
      }
      mPrintParamsStream->printf("    %-35s = %s;\n", paramName, vstr.str().c_str());
      mPrintLuaStream->printf("    %-35s = %s;\n", paramName, vstr.str().c_str());
   }
}

template <typename T>
void PVParams::writeParamArray(const char *paramName, const T *array, int arraysize) {
   if (mPrintParamsStream) {
      pvAssert(mPrintLuaStream != nullptr);
      pvAssert(arraysize >= 0);
      if (arraysize > 0) {
         mPrintParamsStream->printf("    %-35s = [", paramName);
         mPrintLuaStream->printf("    %-35s = {", paramName);
         for (int k = 0; k < arraysize - 1; k++) {
            mPrintParamsStream->printf("%f,", (double)array[k]);
            mPrintLuaStream->printf("%f,", (double)array[k]);
         }
         mPrintParamsStream->printf("%f];\n", (double)array[arraysize - 1]);
         mPrintLuaStream->printf("%f};\n", (double)array[arraysize - 1]);
      }
   }
}

} // end namespace PV

#endif /* PVPARAMS_HPP_ */
