/*
 * PVParams.hpp
 *
 *  Created on: Nov 27, 2008
 *      Author: rasmussn
 */

#ifndef PVPARAMS_HPP_
#define PVPARAMS_HPP_

#include "FileStream.hpp"
#include "fileio.hpp"
#include "include/pv_common.h"
#include "io.hpp"
#include "arch/mpi/mpi.h"
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

   const char *getName() { return mName; }
   double value() {
      mHasBeenReadFlag = true;
      return mParamDblValue;
   }
   const float *valuePtr() {
      mHasBeenReadFlag = true;
      return &mParamValue;
   }
   const double *valueDblPtr() {
      mHasBeenReadFlag = true;
      return &mParamDblValue;
   }
   bool hasBeenRead() { return mHasBeenReadFlag; }
   void clearHasBeenRead() { mHasBeenReadFlag = false; }
   void setValue(double v) {
      mParamValue    = (float)v;
      mParamDblValue = v;
   }
   Parameter *copyParameter() { return new Parameter(mName, mParamDblValue); }

  private:
   char *mName;
   float mParamValue;
   double mParamDblValue;
   bool mHasBeenReadFlag;
};

class ParameterArray {
  public:
   ParameterArray(int initialSize);
   virtual ~ParameterArray();
   int getArraySize() { return mArraySize; }
   const char *getName() { return mName; }
   int setName(const char *name);
   const float *getValues(int *sz) {
      mHasBeenReadFlag = true;
      *sz             = mArraySize;
      return mValues;
   }
   const double *getValuesDbl(int *sz) {
      mHasBeenReadFlag = true;
      *sz             = mArraySize;
      return mValuesDbl;
   }
   int pushValue(double value);
   void resetArraySize() { mArraySize = 0; }
   bool hasBeenRead() { return mHasBeenReadFlag; }
   void clearHasBeenRead() { mHasBeenReadFlag = false; }
   double peek(int index) { return mValuesDbl[index]; }
   ParameterArray *copyParameterArray();

  private:
   bool mNameSet;
   char *mName;
   int mArraySize; // The number of values that have been pushed
   int mBufferSize; // The size of the buffer in memory
   double *mValuesDbl;
   float *mValues;
   bool mHasBeenReadFlag;
};

class ParameterString {
  public:
   ParameterString(const char *name, const char *value);
   virtual ~ParameterString();

   const char *getName() { return mName; }
   const char *getValue() {
      mHasBeenReadFlag = true;
      return mParamValue;
   }
   bool hasBeenRead() { return mHasBeenReadFlag; }
   void clearHasBeenRead() { mHasBeenReadFlag = false; }
   void setValue(const char *s) {
      free(mParamValue);
      mParamValue = s ? strdup(s) : nullptr;
   }
   ParameterString *copyParameterString() { return new ParameterString(mName, mParamValue); }

  private:
   char *mName;
   char *mParamValue;
   bool mHasBeenReadFlag;
};

class ParameterStack {
  public:
   ParameterStack(int maxCount);
   virtual ~ParameterStack();

   int push(Parameter *param);
   Parameter *pop();
   Parameter *peek(int index) { return mParameters[index]; }
   int size() { return mCount; }

  private:
   int mCount;
   int mMaxCount;
   Parameter **mParameters;
};

class ParameterArrayStack {
  public:
   ParameterArrayStack(int initialCount);
   virtual ~ParameterArrayStack();
   int push(ParameterArray *array);
   int size() { return mCount; }
   ParameterArray *peek(int index) {
      return index >= 0 && index < mCount ? mParameterArrays[index] : nullptr;
   }

  private:
   int mCount; // Number of ParameterArrays
   int mAllocation; // Size of buffer
   ParameterArray **mParameterArrays;
};

class ParameterStringStack {
  public:
   ParameterStringStack(int initialCount);
   virtual ~ParameterStringStack();

   int push(ParameterString *param);
   ParameterString *pop();
   ParameterString *peek(int index) {
      return index >= 0 && index < mCount ? mParameterStrings[index] : nullptr;
   }
   int size() { return mCount; }
   const char *lookup(const char *targetname);

  private:
   int mCount;
   int mAllocation;
   ParameterString **mParameterStrings;
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

   const char *getName() { return mName; }
   const char *getGroupKeyword() { return mGroupKeyword; }
   int setGroupKeyword(const char *keyword);
   int setStringStack(ParameterStringStack *stringStack);
   int present(const char *name);
   double value(const char *name);
   bool arrayPresent(const char *name);
   const float *arrayValues(const char *name, int *size);
   const double *arrayValuesDbl(const char *name, int *size);
   int stringPresent(const char *stringName);
   const char *stringValue(const char *stringName);

   /**
    * lookForUnread() tests each parameter in the parameter group for whether it's been read, and
    * prints a message for each unread parameter.  If errorOnUnread is true, the message is an
    * error; otherwise it is a warning. Returns PV_SUCCESS if all parameters have been read,
    * and PV_FAILURE otherwise.
    */
   int lookForUnread(bool errorOnUnread);
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
   char *mName;
   char *mGroupKeyword;
   ParameterStack *mStack;
   ParameterArrayStack *mArrayStack;
   ParameterStringStack *mStringStack;
   int mProcessRank;
};

enum SweepType { SWEEP_UNDEF = 0, SWEEP_NUMBER = 1, SWEEP_STRING = 2 };

class ParameterSweep {
  public:
   ParameterSweep();
   virtual ~ParameterSweep();

   int setGroupAndParameter(const char *groupname, const char *paramname);
   int pushNumericValue(double val);
   int pushStringValue(const char *sval);
   int getNumValues() { return mNumValues; }
   SweepType getType() { return mType; }
   int getNumericValue(int n, double *val);
   const char *getStringValue(int n);
   const char *getGroupName() { return mGroupName; }
   const char *getParamName() { return mParamName; }

  private:
   char *mGroupName;
   char *mParamName;
   SweepType mType;
   int mNumValues;
   int mCurrentBufferSize;
   double *mValuesNumber;
   char **mValuesString;
};

class PVParams {
  public:
   PVParams(size_t initialSize, MPI_Comm mpiComm);
   PVParams(const char *filename, size_t initialSize, MPI_Comm mpiComm);
   PVParams(
         const char *buffer,
         long int bufferLength,
         size_t initialSize,
         MPI_Comm mpiComm);
   virtual ~PVParams();

   bool getParseStatus() { return mParseStatus; }

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
   double value(
         const char *groupName,
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

   /**
    * lookForUnread() tests each parameter in each parameter group for whether it's been read, and
    * prints a message for each unread parameter.  If errorOnUnread is true, the message is an
    * error; otherwise it is a warning. Returns PV_SUCCESS if all parameters have been read,
    * and PV_FAILURE otherwise.
    */
   int lookForUnread(bool errorOnUnread);
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

   /**
    * Randomly shuffles the vector of pointers to the ParameterGroup objects.
    * Used for debugging purposes, to help identify cases where behavior depends
    * on the order of objects in the params file during debugging.
    * The shuffling here has no effect on the RNGs managed by the HyPerCol and used
    * by layers or connections.
    */
   void shuffleGroups(unsigned int seed);

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

   int getNumGroups() { return (int)mGroups.size(); }
   int getNumParamSweeps() { return mNumParamSweeps; }
   int getParameterSweepSize() { return mParameterSweepSize; }
   FileStream *getPrintParamsStream() { return mPrintParamsStream; }
   FileStream *getPrintLuaStream() { return mPrintLuaStream; }

  private:
   int mParseStatus;
   std::vector<ParameterGroup *> mGroups;
   ParameterStack *mStack;
   ParameterArrayStack *mArrayStack;
   ParameterStringStack *mStringStack;
   bool mDebugParsing;
   bool mDisable;
   MPI_Comm mMPIComm;
   int mWorldRank;
   int mWorldSize;

   ParameterArray *mCurrentParamArray;

   int mNumParamSweeps; // The number of different parameters that are changed during the sweep.
   ParameterSweep **mParamSweeps;
   ParameterSweep *mActiveParamSweep;
   int mParameterSweepSize; // The number of parameter value sets in the sweep.  Each ParameterSweep
   // group in the params file must contain the same number of values, which is sweepSize.

   char *mCurrGroupKeyword;
   char *mCurrGroupName;

   char *mCurrSweepGroupName;
   char *mCurrSweepParamName;

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
      const char *groupName,
      const char *paramName,
      T correct_value) {
   int status = PV_SUCCESS;
   if (present(groupName, paramName)) {
      if (mWorldRank == 0) {
         const char *className = groupKeywordFromName(groupName);
         WarnLog().printf(
               "%s \"%s\" does not use parameter %s, but it is present in the parameters file.\n",
               className,
               groupName,
               paramName);
      }
      T params_value = (T)value(
            groupName,
            paramName); // marks param as read so that presentAndNotBeenRead doesn't trip up
      if (params_value != correct_value) {
         status = PV_FAILURE;
         if (mWorldRank == 0) {
            ErrorLog() << "   Value " << params_value << " is inconsistent with correct value "
                       << correct_value << std::endl;
         }
      }
   }
   MPI_Barrier(mMPIComm);
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
         if (paramArrayValue == nullptr) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            Fatal().printf(
                  "%s \"%s\": global rank %d process unable to copy array parameter %s: %s\n",
                  groupKeywordFromName(groupName),
                  groupName,
                  rank,
                  paramName,
                  strerror(errno));
         }
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
