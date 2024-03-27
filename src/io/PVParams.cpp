/*
 * PVParams.cpp
 *
 *  Created on: Nov 27, 2008
 *      Author: rasmussn
 */

#include "PVParams.hpp"
#include "include/pv_common.h"
#include "utils/PVAlloc.hpp"
#include <algorithm> // shuffle, used in shuffleGroups()
#include <assert.h>
#include <climits> // INT_MIN
#include <cmath> // nearbyint()
#include <cstdio>
#include <cstdlib>
#include <cstring> // strcmp(), strcpy()
#include <iostream>
#include <random> // mt19937, used in shuffleGroups()
#ifdef PV_USE_LUA
#include <lua.hpp>
#endif // PV_USE_LUA

#define PARAMETERARRAY_INITIALSIZE 8
#define PARAMETERARRAYSTACK_INITIALCOUNT 5
#define PARAMETERSTRINGSTACK_INITIALCOUNT 5
#define PARAMETERSWEEP_INCREMENTCOUNT 10

// define for debug output
#define DEBUG_PARSING

/**
 * @yyin
 * @action_handler
 * @paramBuffer
 * @len
 */
int pv_parseParameters(PV::PVParams *action_handler, const char *paramBuffer, size_t len);

namespace PV {

/**
 * @name
 * @value
 */
Parameter::Parameter(const char *name, double value) {
   mName       = strdup(name);
   mParamValue      = (float)value;
   mParamDblValue   = value;
   mHasBeenReadFlag = false;
}

Parameter::~Parameter() { free(mName); }

ParameterArray::ParameterArray(int initialSize) {
   mHasBeenReadFlag = false;
   mNameSet         = false;
   mName            = strdup("Unnamed Parameter array");
   mBufferSize      = initialSize;
   mArraySize       = 0;
   mValues          = nullptr;
   if (mBufferSize > 0) {
      mValues    = (float *)calloc(mBufferSize, sizeof(float));
      mValuesDbl = (double *)calloc(mBufferSize, sizeof(double));
      if (mValues == nullptr || mValuesDbl == nullptr) {
         Fatal().printf("ParameterArray failed to allocate memory for \"%s\"\n", getName());
      }
   }
}

ParameterArray::~ParameterArray() {
   free(mName);
   mName = nullptr;
   free(mValues);
   mValues = nullptr;
   free(mValuesDbl);
   mValuesDbl = nullptr;
}

int ParameterArray::setName(const char *name) {
   int status = PV_SUCCESS;
   if (mNameSet == false) {
      free(mName);
      mName    = strdup(name);
      mNameSet = true;
   }
   else {
      ErrorLog().printf(
            "ParameterArray::setName called with \"%s\" but name is already set to \"%s\"\n",
            name,
            mName);
      status = PV_FAILURE;
   }
   return status;
}

int ParameterArray::pushValue(double value) {
   assert(mBufferSize >= mArraySize);
   if (mBufferSize == mArraySize) {
      mBufferSize += PARAMETERARRAY_INITIALSIZE;
      float *new_values = (float *)calloc(mBufferSize, sizeof(float));
      if (new_values == nullptr) {
         Fatal().printf(
               "ParameterArray::pushValue failed to increase array \"%s\" to %d values\n",
               getName(),
               mArraySize + 1);
      }
      memcpy(new_values, mValues, sizeof(float) * mArraySize);
      free(mValues);
      mValues                = new_values;
      double *new_values_dbl = (double *)calloc(mBufferSize, sizeof(double));
      if (new_values == nullptr) {
         Fatal().printf(
               "ParameterArray::pushValue failed to increase array \"%s\" to %d values\n",
               getName(),
               mArraySize + 1);
      }
      memcpy(new_values_dbl, mValuesDbl, sizeof(double) * mArraySize);
      free(mValuesDbl);
      mValuesDbl = new_values_dbl;
   }
   assert(mArraySize < mBufferSize);
   mValuesDbl[mArraySize] = value;
   mValues[mArraySize]    = (float)value;
   mArraySize++;
   return mArraySize;
}

ParameterArray *ParameterArray::copyParameterArray() {
   ParameterArray *returnPA = new ParameterArray(mBufferSize);
   returnPA->setName(mName);
   assert(!std::strcmp(returnPA->getName(), mName));
   for (int i = 0; i < mArraySize; i++) {
      returnPA->pushValue(mValuesDbl[i]);
   }
   return returnPA;
}

/**
 * @name
 * @value
 */
ParameterString::ParameterString(const char *name, const char *value) {
   mName       = name ? strdup(name) : nullptr;
   mParamValue      = value ? strdup(value) : nullptr;
   mHasBeenReadFlag = false;
}

ParameterString::~ParameterString() {
   free(mName);
   free(mParamValue);
}

/**
 * @maxCount
 */
ParameterStack::ParameterStack(int maxCount) {
   mMaxCount   = maxCount;
   mCount      = 0;
   mParameters = (Parameter **)malloc(maxCount * sizeof(Parameter *));
}

ParameterStack::~ParameterStack() {
   for (int i = 0; i < mCount; i++) {
      delete mParameters[i];
   }
   free(mParameters);
}

/**
 * @param
 */
int ParameterStack::push(Parameter *param) {
   assert(mCount < mMaxCount);
   mParameters[mCount++] = param;
   return 0;
}

Parameter *ParameterStack::pop() {
   assert(mCount > 0);
   return mParameters[mCount--];
}

ParameterArrayStack::ParameterArrayStack(int initialCount) {
   mAllocation      = initialCount;
   mCount           = 0;
   mParameterArrays = nullptr;
   if (initialCount > 0) {
      mParameterArrays = (ParameterArray **)calloc(mAllocation, sizeof(ParameterArray *));
      if (mParameterArrays == nullptr) {
         Fatal().printf(
               "ParameterArrayStack unable to allocate %d parameter arrays\n", initialCount);
      }
   }
}

ParameterArrayStack::~ParameterArrayStack() {
   for (int k = 0; k < mCount; k++) {
      delete mParameterArrays[k];
      mParameterArrays[k] = nullptr;
   }
   free(mParameterArrays);
   mParameterArrays = nullptr;
}

int ParameterArrayStack::push(ParameterArray *array) {
   assert(mCount <= mAllocation);
   if (mCount == mAllocation) {
      int newallocation = mAllocation + RESIZE_ARRAY_INCR;
      ParameterArray **newParameterArrays =
            (ParameterArray **)malloc(newallocation * sizeof(ParameterArray *));
      if (!newParameterArrays)
         return PV_FAILURE;
      for (int i = 0; i < mCount; i++) {
         newParameterArrays[i] = mParameterArrays[i];
      }
      mAllocation = newallocation;
      free(mParameterArrays);
      mParameterArrays = newParameterArrays;
   }
   assert(mCount < mAllocation);
   mParameterArrays[mCount] = array;
   mCount++;
   return PV_SUCCESS;
}

/*
 * initialCount
 */
ParameterStringStack::ParameterStringStack(int initialCount) {
   mAllocation       = initialCount;
   mCount            = 0;
   mParameterStrings = (ParameterString **)calloc(mAllocation, sizeof(ParameterString *));
}

ParameterStringStack::~ParameterStringStack() {
   for (int i = 0; i < mCount; i++) {
      delete mParameterStrings[i];
   }
   free(mParameterStrings);
}

/*
 * @param
 */
int ParameterStringStack::push(ParameterString *param) {
   assert(mCount <= mAllocation);
   if (mCount == mAllocation) {
      int newallocation = mAllocation + RESIZE_ARRAY_INCR;
      ParameterString **newparameterStrings =
            (ParameterString **)malloc(newallocation * sizeof(ParameterString *));
      if (!newparameterStrings)
         return PV_FAILURE;
      for (int i = 0; i < mCount; i++) {
         newparameterStrings[i] = mParameterStrings[i];
      }
      mAllocation = newallocation;
      free(mParameterStrings);
      mParameterStrings = newparameterStrings;
   }
   assert(mCount < mAllocation);
   mParameterStrings[mCount++] = param;
   return PV_SUCCESS;
}

ParameterString *ParameterStringStack::pop() {
   if (mCount > 0) {
      return mParameterStrings[mCount--];
   }
   else
      return nullptr;
}

const char *ParameterStringStack::lookup(const char *targetname) {
   const char *result = nullptr;
   for (int i = 0; i < mCount; i++) {
      if (!std::strcmp(mParameterStrings[i]->getName(), targetname)) {
         result = mParameterStrings[i]->getValue();
      }
   }
   return result;
}

/**
 * @name
 * @stack
 * @string_stack
 * @rank
 */
ParameterGroup::ParameterGroup(
      char *name,
      ParameterStack *stack,
      ParameterArrayStack *array_stack,
      ParameterStringStack *string_stack,
      int rank) {
   mName         = strdup(name);
   mGroupKeyword = nullptr;
   mStack        = stack;
   mArrayStack   = array_stack;
   mStringStack  = string_stack;
   mProcessRank  = rank;
}

ParameterGroup::~ParameterGroup() {
   free(mName);
   mName = nullptr;
   free(mGroupKeyword);
   mGroupKeyword = nullptr;
   delete mStack;
   mStack = nullptr;
   delete mArrayStack;
   mArrayStack = nullptr;
   delete mStringStack;
   mStringStack = nullptr;
}

int ParameterGroup::setGroupKeyword(const char *keyword) {
   if (mGroupKeyword == nullptr) {
      size_t keywordlen = strlen(keyword);
      mGroupKeyword      = (char *)malloc(keywordlen + 1);
      if (mGroupKeyword) {
         std::strcpy(mGroupKeyword, keyword);
      }
   }
   return mGroupKeyword == nullptr ? PV_FAILURE : PV_SUCCESS;
}

int ParameterGroup::setStringStack(ParameterStringStack *stringStack) {
   mStringStack = stringStack;
   // ParameterGroup::setStringStack takes ownership of the stringStack;
   // i.e. it will delete it when the ParameterGroup is deleted.
   // You shouldn't use a stringStack after calling this routine with it.
   // Instead, query it with ParameterGroup::stringPresent and
   // ParameterGroup::stringValue methods.
   return mStringStack == nullptr ? PV_FAILURE : PV_SUCCESS;
}

/**
 * @name
 */
int ParameterGroup::present(const char *name) {
   int count = mStack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = mStack->peek(i);
      if (std::strcmp(name, p->getName()) == 0) {
         return 1; // string is present
      }
   }
   return 0; // string not present
}

/**
 * @name
 */
double ParameterGroup::value(const char *name) {
   int count = mStack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = mStack->peek(i);
      if (std::strcmp(name, p->getName()) == 0) {
         return p->value();
      }
   }
   Fatal().printf(
         "PVParams::ParameterGroup::value: ERROR, couldn't find a value for %s"
         " in group %s\n",
         name,
         mName);
   return PV_FAILURE; // suppresses warning in compilers that don't recognize Fatal always exits.
}

bool ParameterGroup::arrayPresent(const char *name) {
   bool arrayFound = false;
   int count        = mArrayStack->size();
   for (int i = 0; i < count; i++) {
      ParameterArray *p = mArrayStack->peek(i);
      if (std::strcmp(name, p->getName()) == 0) {
         arrayFound = true; // string is present
         break;
      }
   }
   if (!arrayFound) {
      arrayFound = (present(name) != 0);
   }
   return arrayFound;
}

const float *ParameterGroup::arrayValues(const char *name, int *size) {
   int count         = mArrayStack->size();
   *size             = 0;
   const float *v    = nullptr;
   ParameterArray *p = nullptr;
   for (int i = 0; i < count; i++) {
      p = mArrayStack->peek(i);
      if (std::strcmp(name, p->getName()) == 0) {
         v = p->getValues(size);
         break;
      }
   }
   if (!v) {
      Parameter *q = nullptr;
      for (int i = 0; i < mStack->size(); i++) {
         Parameter *q1 = mStack->peek(i);
         assert(q1);
         if (std::strcmp(name, q1->getName()) == 0) {
            q = q1;
            break;
         }
      }
      if (q) {
         v     = q->valuePtr();
         *size = 1;
      }
   }
   return v;
}

const double *ParameterGroup::arrayValuesDbl(const char *name, int *size) {
   int count         = mArrayStack->size();
   *size             = 0;
   const double *v   = nullptr;
   ParameterArray *p = nullptr;
   for (int i = 0; i < count; i++) {
      p = mArrayStack->peek(i);
      if (std::strcmp(name, p->getName()) == 0) {
         v = p->getValuesDbl(size);
         break;
      }
   }
   if (!v) {
      Parameter *q = nullptr;
      for (int i = 0; i < mStack->size(); i++) {
         Parameter *q1 = mStack->peek(i);
         assert(q1);
         if (std::strcmp(name, q1->getName()) == 0) {
            q = q1;
            break;
         }
      }
      if (q) {
         v     = q->valueDblPtr();
         *size = 1;
      }
   }
   return v;
}

int ParameterGroup::stringPresent(const char *stringName) {
   // not really necessary, as stringValue returns nullptr if the
   // string is not found, but included on the analogy with
   // value and present methods for floating-point parameters
   if (!stringName)
      return 0;
   int count = mStringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *pstr = mStringStack->peek(i);
      assert(pstr);
      if (!std::strcmp(stringName, pstr->getName())) {
         return 1; // string is present
      }
   }
   return 0; // string not present
}

const char *ParameterGroup::stringValue(const char *stringName) {
   if (!stringName)
      return nullptr;
   int count = mStringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *pstr = mStringStack->peek(i);
      assert(pstr);
      if (!std::strcmp(stringName, pstr->getName())) {
         return pstr->getValue();
      }
   }
   return nullptr;
}

int ParameterGroup::lookForUnread(bool errorOnUnread) {
   int status = PV_SUCCESS;
   int count;
   count = mStack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = mStack->peek(i);
      if (!p->hasBeenRead()) {
         if (mProcessRank == 0) {
            std::string message("Parameter group \"#1\": parameter \"#2\" has not been read.\n");
            message.replace(message.find("#1"), 2, getName());
            message.replace(message.find("#2"), 2, p->getName());
            if (errorOnUnread) {
               ErrorLog() << message;
            }
            else {
               WarnLog() << message;
            }
         }
         status = PV_FAILURE;
      }
   }
   count = mArrayStack->size();
   for (int i = 0; i < count; i++) {
      ParameterArray *parr = mArrayStack->peek(i);
      if (!parr->hasBeenRead()) {
         if (mProcessRank == 0) {
            std::string message(
                  "Parameter group \"#1\": array parameter \"#2\" has not been read.\n");
            message.replace(message.find("#1"), 2, getName());
            message.replace(message.find("#2"), 2, parr->getName());
            if (errorOnUnread) {
               ErrorLog() << message;
            }
            else {
               WarnLog() << message;
            }
         }
      }
   }
   count = mStringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *pstr = mStringStack->peek(i);
      if (!pstr->hasBeenRead()) {
         if (mProcessRank == 0) {
            std::string message(
                  "Parameter group \"#1\": string parameter \"#2\" has not been read.\n");
            message.replace(message.find("#1"), 2, getName());
            message.replace(message.find("#2"), 2, pstr->getName());
            if (errorOnUnread) {
               ErrorLog() << message;
            }
            else {
               WarnLog() << message;
            }
         }
         status = PV_FAILURE;
      }
   }
   return status;
}

bool ParameterGroup::hasBeenRead(const char *paramName) {
   int count;
   count = mStack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = mStack->peek(i);
      if (!std::strcmp(p->getName(), paramName)) {
         return p->hasBeenRead();
      }
   }
   count = mArrayStack->size();
   for (int i = 0; i < count; i++) {
      ParameterArray *parr = mArrayStack->peek(i);
      if (!std::strcmp(parr->getName(), paramName)) {
         return parr->hasBeenRead();
      }
   }
   count = mStringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *pstr = mStringStack->peek(i);
      if (!std::strcmp(pstr->getName(), paramName)) {
         return pstr->hasBeenRead();
      }
   }
   return false;
}

int ParameterGroup::clearHasBeenReadFlags() {
   int status = PV_SUCCESS;
   int count  = mStack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = mStack->peek(i);
      p->clearHasBeenRead();
   }
   count = mArrayStack->size();
   for (int i = 0; i < count; i++) {
      ParameterArray *parr = mArrayStack->peek(i);
      parr->clearHasBeenRead();
   }
   count = mStringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *pstr = mStringStack->peek(i);
      pstr->clearHasBeenRead();
   }
   return status;
}

int ParameterGroup::pushNumerical(Parameter *param) { return mStack->push(param); }

int ParameterGroup::pushString(ParameterString *param) { return mStringStack->push(param); }

int ParameterGroup::setValue(const char *param_name, double value) {
   int status = PV_SUCCESS;
   int count  = mStack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = mStack->peek(i);
      if (std::strcmp(param_name, p->getName()) == 0) {
         p->setValue(value);
         return PV_SUCCESS;
      }
   }
   Fatal().printf(
         "PVParams::ParameterGroup::setValue: ERROR, couldn't find parameter %s"
         " in group \"%s\"\n",
         param_name,
         getName());

   return status;
}

int ParameterGroup::setStringValue(const char *param_name, const char *svalue) {
   int status = PV_SUCCESS;
   int count  = mStringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *p = mStringStack->peek(i);
      if (std::strcmp(param_name, p->getName()) == 0) {
         p->setValue(svalue);
         return PV_SUCCESS;
      }
   }
   Fatal().printf(
         "PVParams::ParameterGroup::setStringValue: ERROR, couldn't find a string value for %s"
         " in group \"%s\"\n",
         param_name,
         getName());

   return status;
}

/**
 * A function to return a copy of the parameter group's stack.
 */
ParameterStack *ParameterGroup::copyStack() {
   ParameterStack *returnStack = new ParameterStack(MAX_PARAMS);
   for (int i = 0; i < mStack->size(); i++) {
      returnStack->push(mStack->peek(i)->copyParameter());
   }
   return returnStack;
}

/**
 * A function to return a copy of the parameter group's stack.
 */
ParameterArrayStack *ParameterGroup::copyArrayStack() {
   ParameterArrayStack *returnStack = new ParameterArrayStack(PARAMETERARRAYSTACK_INITIALCOUNT);
   for (int i = 0; i < mArrayStack->size(); i++) {
      returnStack->push(mArrayStack->peek(i)->copyParameterArray());
   }
   return returnStack;
}

/**
 * A function to return a copy of the parameter group's stack.
 */
ParameterStringStack *ParameterGroup::copyStringStack() {
   ParameterStringStack *returnStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);
   for (int i = 0; i < mStringStack->size(); i++) {
      returnStack->push(mStringStack->peek(i)->copyParameterString());
   }
   return returnStack;
}

ParameterSweep::ParameterSweep() {
   mGroupName         = nullptr;
   mParamName         = nullptr;
   mNumValues         = 0;
   mCurrentBufferSize = 0;
   mType              = SWEEP_UNDEF;
   mValuesNumber      = nullptr;
   mValuesString      = nullptr;
}

ParameterSweep::~ParameterSweep() {
   free(mGroupName);
   mGroupName = nullptr;
   free(mParamName);
   mParamName = nullptr;
   free(mValuesNumber);
   mValuesNumber = nullptr;
   if (mValuesString != nullptr) {
      for (int k = 0; k < mNumValues; k++) {
         free(mValuesString[k]);
      }
      free(mValuesString);
      mValuesString = nullptr;
   }
}

int ParameterSweep::setGroupAndParameter(const char *groupname, const char *parametername) {
   int status = PV_SUCCESS;
   if (mGroupName != nullptr || mParamName != nullptr) {
      ErrorLog(errorMessage);
      errorMessage.printf("ParameterSweep::setGroupParameter: ");
      if (mGroupName != nullptr) {
         errorMessage.printf(" group name has already been set to \"%s\".", mGroupName);
      }
      if (mParamName != nullptr) {
         errorMessage.printf(" param name has already been set to \"%s\".", mParamName);
      }
      errorMessage.printf("\n");
      status = PV_FAILURE;
   }
   else {
      mGroupName = strdup(groupname);
      mParamName = strdup(parametername);
      // Check for duplicates
   }
   return status;
}

int ParameterSweep::pushNumericValue(double val) {
   int status = PV_SUCCESS;
   if (mNumValues == 0) {
      mType = SWEEP_NUMBER;
   }
   assert(mType == SWEEP_NUMBER);
   assert(mValuesString == nullptr);

   assert(mNumValues <= mCurrentBufferSize);
   if (mNumValues == mCurrentBufferSize) {
      mCurrentBufferSize += PARAMETERSWEEP_INCREMENTCOUNT;
      double *newValuesNumber = (double *)calloc(mCurrentBufferSize, sizeof(double));
      if (newValuesNumber == nullptr) {
         ErrorLog().printf("ParameterSweep:pushNumericValue: unable to allocate memory\n");
         status = PV_FAILURE;
         abort();
      }
      for (int k = 0; k < mNumValues; k++) {
         newValuesNumber[k] = mValuesNumber[k];
      }
      free(mValuesNumber);
      mValuesNumber = newValuesNumber;
   }
   mValuesNumber[mNumValues] = val;
   mNumValues++;
   return status;
}

int ParameterSweep::pushStringValue(const char *sval) {
   int status = PV_SUCCESS;
   if (mNumValues == 0) {
      mType = SWEEP_STRING;
   }
   assert(mType == SWEEP_STRING);
   assert(mValuesNumber == nullptr);

   assert(mNumValues <= mCurrentBufferSize);
   if (mNumValues == mCurrentBufferSize) {
      mCurrentBufferSize += PARAMETERSWEEP_INCREMENTCOUNT;
      char **newValuesString = (char **)calloc(mCurrentBufferSize, sizeof(char *));
      if (newValuesString == nullptr) {
         ErrorLog().printf("ParameterSweep:pushStringValue: unable to allocate memory\n");
         status = PV_FAILURE;
         abort();
      }
      for (int k = 0; k < mNumValues; k++) {
         newValuesString[k] = mValuesString[k];
      }
      free(mValuesString);
      mValuesString = newValuesString;
   }
   mValuesString[mNumValues] = strdup(sval);
   mNumValues++;
   return status;
}

int ParameterSweep::getNumericValue(int n, double *val) {
   int status = PV_SUCCESS;
   assert(mValuesNumber != nullptr);
   if (mType != SWEEP_NUMBER || n < 0 || n >= mNumValues) {
      status = PV_FAILURE;
   }
   else {
      *val = mValuesNumber[n];
   }
   return status;
}

const char *ParameterSweep::getStringValue(int n) {
   char *str = nullptr;
   assert(mValuesString != nullptr);
   if (mType == SWEEP_STRING && n >= 0 && n < mNumValues) {
      str = mValuesString[n];
   }
   return str;
}

/**
 * @filename
 * @initialSize
 * @mpiComm
 */
PVParams::PVParams(const char *filename, size_t initialSize, MPI_Comm mpiComm) {
   mMPIComm = mpiComm;
   initialize(initialSize);
   parseFile(filename);
}

/*
 * @initialSize
 * @mpiComm
 */
PVParams::PVParams(size_t initialSize, MPI_Comm mpiComm) {
   mMPIComm = mpiComm;
   initialize(initialSize);
}

/*
 * @buffer
 * @bufferLength
 * @initialSize
 * @mpiComm
 */
PVParams::PVParams(
      const char *buffer,
      long int bufferLength,
      size_t initialSize,
      MPI_Comm mpiComm) {
   mMPIComm = mpiComm;
   initialize(initialSize);
   parseBuffer(buffer, bufferLength);
}

PVParams::~PVParams() {
   for (auto &g : mGroups) {
      delete g;
   }
   delete mCurrentParamArray;
   mCurrentParamArray = nullptr;
   delete mStack;
   delete mArrayStack;
   delete mStringStack;
   delete mActiveParamSweep;
   for (int i = 0; i < mNumParamSweeps; i++) {
      delete mParamSweeps[i];
   }
   free(mParamSweeps);
   mParamSweeps = nullptr;
}

/*
 * @initialSize
 */
int PVParams::initialize(size_t initialSize) {
   // Get world rank and size
   MPI_Comm_rank(mMPIComm, &mWorldRank);
   MPI_Comm_size(mMPIComm, &mWorldSize);

   mGroups.reserve(initialSize);
   mStack       = new ParameterStack(MAX_PARAMS);
   mArrayStack  = new ParameterArrayStack(PARAMETERARRAYSTACK_INITIALCOUNT);
   mStringStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);

   mCurrentParamArray = new ParameterArray(PARAMETERARRAYSTACK_INITIALCOUNT);

   mNumParamSweeps = 0;
   mParamSweeps    = nullptr;
   newActiveParamSweep();
#ifdef DEBUG_PARSING
   mDebugParsing = true;
#else
   mDebugParsing      = false;
#endif // DEBUG_PARSING
   mDisable = false;

   return (mStack && mStringStack && mActiveParamSweep) ? PV_SUCCESS : PV_FAILURE;
}

int PVParams::newActiveParamSweep() {
   int status        = PV_SUCCESS;
   mActiveParamSweep = new ParameterSweep();
   if (mActiveParamSweep == nullptr) {
      Fatal().printf("PVParams::newActiveParamSweep: unable to create new Parameter Sweep");
      status = PV_FAILURE;
   }
   return status;
}

int PVParams::parseFile(const char *filename) {
   int rootproc      = 0;
   char *paramBuffer = nullptr;
   size_t bufferlen;
   if (mWorldRank == rootproc) {
      std::string paramBufferString("");
      loadParamBuffer(filename, paramBufferString);
      bufferlen = paramBufferString.size();
      // Older versions of MPI_Send require void*, not void const*
      paramBuffer = (char *)pvMalloc(bufferlen + 1);
      memcpy(paramBuffer, paramBufferString.c_str(), bufferlen);
      paramBuffer[bufferlen] = '\0';

#ifdef PV_USE_MPI
      int sz = mWorldSize;
      for (int i = 0; i < sz; i++) {
         if (i == rootproc)
            continue;
         MPI_Send(paramBuffer, (int)bufferlen, MPI_CHAR, i, 31, mMPIComm);
      }
#endif // PV_USE_MPI
   }
   else { // rank != rootproc
#ifdef PV_USE_MPI
      MPI_Status mpi_status;
      int count;
      MPI_Probe(rootproc, 31, mMPIComm, &mpi_status);
      // int status =
      MPI_Get_count(&mpi_status, MPI_CHAR, &count);
      bufferlen   = (size_t)count;
      paramBuffer = (char *)malloc(bufferlen);
      if (paramBuffer == nullptr) {
         Fatal().printf(
               "PVParams::parseFile: Rank %d process unable to allocate memory for params buffer\n",
               mWorldRank);
      }
      MPI_Recv(
            paramBuffer,
            (int)bufferlen,
            MPI_CHAR,
            rootproc,
            31,
            mMPIComm,
            MPI_STATUS_IGNORE);
#endif // PV_USE_MPI
   }

   int status = parseBuffer(paramBuffer, bufferlen);
   free(paramBuffer);
   return status;
}

void PVParams::loadParamBuffer(char const *filename, std::string &paramsFileString) {
   if (filename == nullptr) {
      Fatal() << "PVParams::loadParamBuffer: filename is null\n";
   }
   struct stat filestatus;
   if (PV_stat(filename, &filestatus)) {
      Fatal().printf(
            "PVParams::parseFile unable to get status of file \"%s\": %s\n",
            filename,
            strerror(errno));
   }
   if (filestatus.st_mode & S_IFDIR) {
      Fatal().printf("PVParams::parseFile: specified file \"%s\" is a directory.\n", filename);
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

bool PVParams::hasSweepValue(const char *inParamName) {
   bool out = false;
   const char *group_name;
   for (int k = 0; k < getNumParamSweeps(); k++) {
      ParameterSweep *sweep  = mParamSweeps[k];
      group_name             = sweep->getGroupName();
      const char *param_name = sweep->getParamName();
      ParameterGroup *gp     = group(group_name);
      if (gp == nullptr) {
         Fatal().printf(
               "PVParams::hasSweepValue error: ParameterSweep %d (zero-indexed) refers to "
               "non-existent group \"%s\"\n",
               k,
               group_name);
      }
      if (!std::strcmp(gp->getGroupKeyword(), "HyPerCol") &&
          !std::strcmp(param_name, inParamName)) {
         out = true;
         break;
      }
   }
   return out;
}

int PVParams::parseBuffer(char const *buffer, long int bufferLength) {
   // Assumes that each MPI process has the same contents in buffer.

   // This is where it calls the scanner and parser
   mParseStatus = pv_parseParameters(this, buffer, bufferLength);
   if (mParseStatus != 0) {
      ErrorLog().printf(
            "Rank %d process: pv_parseParameters failed with return value %d\n",
            mWorldRank,
            mParseStatus);
   }
   getOutputStream().flush();

   setParameterSweepSize(); // Need to set sweepSize here, because if the outputPath sweep needs to
   // be created we need to know the size.

   // If there is at least one ParameterSweep  and none of them set outputPath, create a
   // parameterSweep that does set outputPath.

   // If both parameterSweep and batchSweep is set, must autoset output path, as there is no way to
   // specify both paramSweep and batchSweep
   if (getNumParamSweeps() > 0) {
      if (!hasSweepValue("outputPath")) {
         const char *hypercolgroupname = nullptr;
         const char *outputPathName    = nullptr;
         for (auto &g : mGroups) {
            if (g->getGroupKeyword(), "HyPerCol") {
               hypercolgroupname = g->getName();
               outputPathName    = g->stringValue("outputPath");
               if (outputPathName == nullptr) {
                  Fatal().printf(
                        "PVParams::outputPath must be specified if parameterSweep does "
                        "not sweep over outputPath\n");
               }
               break;
            }
         }
         if (hypercolgroupname == nullptr) {
            ErrorLog().printf("PVParams::parseBuffer: no HyPerCol group\n");
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
            mActiveParamSweep->pushStringValue(outputPathStr.c_str());
         }
         addActiveParamSweep(hypercolgroupname, "outputPath");
      }

      if (!hasSweepValue("checkpointWriteDir")) {
         const char *hypercolgroupname  = nullptr;
         const char *checkpointWriteDir = nullptr;
         for (auto &g : mGroups) {
            if (g->getGroupKeyword(), "HyPerCol") {
               hypercolgroupname  = g->getName();
               checkpointWriteDir = g->stringValue("checkpointWriteDir");
               // checkpointWriteDir can be nullptr if checkpointWrite is set to false
               break;
            }
         }
         if (hypercolgroupname == nullptr) {
            ErrorLog().printf("PVParams::parseBuffer: no HyPerCol group\n");
            abort();
         }
         if (checkpointWriteDir) {
            // Push the strings "[checkpointWriteDir]/paramsweep_[n]/"
            // to the parameter sweep, where [n] ranges from 0 to mParameterSweepSize - 1,
            // and is zero-padded so that the parameter sweep's checkpointWriteDir directories
            // sort the same lexicographically and numerically.
            auto lenmax = std::to_string(mParameterSweepSize - 1).size();
            for (int i = 0; i < mParameterSweepSize; i++) {
               std::string checkpointWriteDirStr(checkpointWriteDir);
               checkpointWriteDirStr.append("/paramsweep_");
               std::string serialNumberStr = std::to_string(i);
               auto len                    = serialNumberStr.size();
               if (len < lenmax) {
                  checkpointWriteDirStr.append(lenmax - len, '0');
               }
               checkpointWriteDirStr.append(serialNumberStr);
               checkpointWriteDirStr.append("/");
               mActiveParamSweep->pushStringValue(checkpointWriteDirStr.c_str());
            }
            addActiveParamSweep(hypercolgroupname, "checkpointWriteDir");
         }
      }
   }

   // Each ParameterSweep needs to have its group/parameter pair added to the database, if it's not
   // already present.
   for (int k = 0; k < getNumParamSweeps(); k++) {
      ParameterSweep *sweep  = mParamSweeps[k];
      const char *group_name = sweep->getGroupName();
      const char *param_name = sweep->getParamName();
      SweepType type         = sweep->getType();
      ParameterGroup *g      = group(group_name);
      if (g == nullptr) {
         ErrorLog().printf("ParameterSweep: there is no group \"%s\"\n", group_name);
         abort();
      }
      switch (type) {
         case SWEEP_NUMBER:
            if (!g->present(param_name)) {
               Parameter *p = new Parameter(param_name, 0.0);
               g->pushNumerical(p);
            }
            break;
         case SWEEP_STRING:
            if (!g->stringPresent(param_name)) {
               ParameterString *p = new ParameterString(param_name, "");
               g->pushString(p);
            }
            break;
         default: assert(0); break;
      }
   }

   clearHasBeenReadFlags();

   return PV_SUCCESS;
}

// TODO other integer types should also use valueInt
template <>
void PVParams::ioParamValue<int>(
      enum ParamsIOFlag ioFlag,
      const char *groupName,
      const char *paramName,
      int *paramValue,
      int defaultValue,
      bool warnIfAbsent) {
   switch (ioFlag) {
      case PARAMS_IO_READ:
         *paramValue = valueInt(groupName, paramName, defaultValue, warnIfAbsent);
         break;
      case PARAMS_IO_WRITE: writeParam(paramName, *paramValue); break;
   }
}

template <>
void PVParams::ioParamValueRequired<int>(
      enum ParamsIOFlag ioFlag,
      const char *groupName,
      const char *paramName,
      int *paramValue) {
   switch (ioFlag) {
      case PARAMS_IO_READ: *paramValue = valueInt(groupName, paramName); break;
      case PARAMS_IO_WRITE: writeParam(paramName, *paramValue); break;
   }
}

int PVParams::setParameterSweepSize() {
   mParameterSweepSize = -1;
   for (int k = 0; k < this->getNumParamSweeps(); k++) {
      if (mParameterSweepSize < 0) {
         mParameterSweepSize = this->mParamSweeps[k]->getNumValues();
      }
      else {
         if (mParameterSweepSize != this->mParamSweeps[k]->getNumValues()) {
            ErrorLog().printf(
                  "PVParams::setParameterSweepSize: all ParameterSweeps in the "
                  "parameters file must have the same number of entries.\n");
            abort();
         }
      }
   }
   if (mParameterSweepSize < 0)
      mParameterSweepSize = 0;
   return mParameterSweepSize;
}

int PVParams::setParameterSweepValues(int n) {
   int status = PV_SUCCESS;
   // Set parameter sweeps
   if (n < 0 || n >= mParameterSweepSize) {
      status = PV_FAILURE;
      return status;
   }
   for (int k = 0; k < this->getNumParamSweeps(); k++) {
      ParameterSweep *paramSweep = mParamSweeps[k];
      SweepType type             = paramSweep->getType();
      const char *group_name     = paramSweep->getGroupName();
      const char *param_name     = paramSweep->getParamName();
      ParameterGroup *gp         = group(group_name);
      assert(gp != nullptr);

      const char *s;
      double v = 0.0f;
      switch (type) {
         case SWEEP_NUMBER:
            paramSweep->getNumericValue(n, &v);
            gp->setValue(param_name, v);
            break;
         case SWEEP_STRING:
            s = paramSweep->getStringValue(n);
            gp->setStringValue(param_name, s);
            break;
         default: assert(0); break;
      }
   }
   return status;
}

/**
 * @groupName
 * @paramName
 */
int PVParams::present(const char *groupName, const char *paramName) {
   ParameterGroup *g = group(groupName);
   if (g == nullptr) {
      if (mWorldRank == 0) {
         ErrorLog().printf("PVParams::present: couldn't find a group for %s\n", groupName);
      }
      exit(EXIT_FAILURE);
   }

   return g->present(paramName);
}

/**
 * @groupName
 * @paramName
 */
double PVParams::value(const char *groupName, const char *paramName) {
   ParameterGroup *g = group(groupName);
   if (g == nullptr) {
      if (mWorldRank == 0) {
         ErrorLog().printf("PVParams::value: ERROR, couldn't find a group for %s\n", groupName);
      }
      exit(EXIT_FAILURE);
   }

   return g->value(paramName);
}

int PVParams::valueInt(const char *groupName, const char *paramName) {
   double v = value(groupName, paramName);
   return convertParamToInt(v);
}

int PVParams::valueInt(
      const char *groupName,
      const char *paramName,
      int initialValue,
      bool warnIfAbsent) {
   double v = value(groupName, paramName, (double)initialValue, warnIfAbsent);
   return convertParamToInt(v);
}

int PVParams::convertParamToInt(double value) {
   int y = 0;
   if (value >= (double)INT_MAX) {
      y = INT_MAX;
   }
   else if (value <= (double)INT_MIN) {
      y = INT_MIN;
   }
   else {
      y = (int)nearbyint(value);
   }
   return y;
}

/**
 * @groupName
 * @paramName
 * @initialValue
 */
double PVParams::value(
      const char *groupName,
      const char *paramName,
      double initialValue,
      bool warnIfAbsent) {
   if (present(groupName, paramName)) {
      return value(groupName, paramName);
   }
   else {
      if (warnIfAbsent && mWorldRank == 0) {
         WarnLog().printf(
               "Using default value %f for parameter \"%s\" in group \"%s\"\n",
               initialValue,
               paramName,
               groupName);
      }
      return initialValue;
   }
}

template <>
void PVParams::writeParam<bool>(const char *paramName, bool paramValue) {
   if (mPrintParamsStream != nullptr) {
      pvAssert(mPrintLuaStream);
      std::stringstream vstr("");
      vstr << (paramValue ? "true" : "false");
      mPrintParamsStream->printf("    %-35s = %s;\n", paramName, vstr.str().c_str());
      mPrintLuaStream->printf("    %-35s = %s;\n", paramName, vstr.str().c_str());
   }
}

bool PVParams::arrayPresent(const char *groupName, const char *paramName) {
   ParameterGroup *g = group(groupName);
   if (g == nullptr) {
      if (mWorldRank == 0) {
         ErrorLog().printf("PVParams::present: couldn't find a group for %s\n", groupName);
      }
      exit(EXIT_FAILURE);
   }

   return g->arrayPresent(paramName);
}

// Could use a template function for arrayValues and arrayValuesDbl
/*
 *  @groupName
 *  @paramName
 *  @size
 */
const float *
PVParams::arrayValues(const char *groupName, const char *paramName, int *size, bool warnIfAbsent) {
   ParameterGroup *g = group(groupName);
   if (g == nullptr) {
      if (mWorldRank == 0) {
         ErrorLog().printf("PVParams::value: couldn't find a group for %s\n", groupName);
      }
      return nullptr;
   }
   const float *retval = g->arrayValues(paramName, size);
   if (retval == nullptr) {
      assert(*size == 0);
      if (mWorldRank == 0) {
         WarnLog().printf(
               "Using empty array for parameter \"%s\" in group \"%s\"\n", paramName, groupName);
      }
   }
   return retval;
}

/*
 *  @groupName
 *  @paramName
 *  @size
 */
const double *PVParams::arrayValuesDbl(
      const char *groupName,
      const char *paramName,
      int *size,
      bool warnIfAbsent) {
   ParameterGroup *g = group(groupName);
   if (g == nullptr) {
      if (mWorldRank == 0) {
         ErrorLog().printf("PVParams::value: couldn't find a group for %s\n", groupName);
      }
      return nullptr;
   }
   const double *retval = g->arrayValuesDbl(paramName, size);
   if (retval == nullptr) {
      assert(*size == 0);
      if (mWorldRank == 0) {
         WarnLog().printf(
               "Using empty array for parameter \"%s\" in group \"%s\"\n", paramName, groupName);
      }
   }
   return retval;
}

void PVParams::ioParamString(
      enum ParamsIOFlag ioFlag,
      const char *groupName,
      const char *paramName,
      char **paramStringValue,
      const char *defaultValue,
      bool warnIfAbsent) {
   const char *paramString = nullptr;
   switch (ioFlag) {
      case PARAMS_IO_READ:
         if (stringPresent(groupName, paramName)) {
            paramString = stringValue(groupName, paramName, warnIfAbsent);
         }
         else {
            // parameter was not set in params file; use the default.  But default might or might
            // not be nullptr.
            if (mWorldRank == 0 and warnIfAbsent == true) {
               if (defaultValue != nullptr) {
                  WarnLog().printf(
                        "Using default value \"%s\" for string parameter \"%s\" in group \"%s\"\n",
                        defaultValue,
                        paramName,
                        groupName);
               }
               else {
                  WarnLog().printf(
                        "Using default value of nullptr for string parameter \"%s\" in group "
                        "\"%s\"\n",
                        paramName,
                        groupName);
               }
            }
            paramString = defaultValue;
         }
         if (paramString != nullptr) {
            *paramStringValue = strdup(paramString);
            FatalIf(
                  *paramStringValue == nullptr,
                  "Global rank %d process unable to copy param %s in group \"%s\": %s\n",
                  mWorldRank,
                  paramName,
                  groupName,
                  strerror(errno));
         }
         else {
            *paramStringValue = nullptr;
         }
         break;
      case PARAMS_IO_WRITE: writeParamString(paramName, *paramStringValue);
   }
}

void PVParams::ioParamStringRequired(
      enum ParamsIOFlag ioFlag,
      const char *groupName,
      const char *paramName,
      char **paramStringValue) {
   const char *paramString = nullptr;
   switch (ioFlag) {
      case PARAMS_IO_READ:
         paramString = stringValue(groupName, paramName, false /*warnIfAbsent*/);
         if (paramString != nullptr) {
            *paramStringValue = strdup(paramString);
            FatalIf(
                  *paramStringValue == nullptr,
                  "Global Rank %d process unable to copy param %s in group \"%s\": %s\n",
                  mWorldRank,
                  paramName,
                  groupName,
                  strerror(errno));
         }
         else if (!stringPresent(groupName, paramName)) {
            // Setting the param to NULL explicitly is allowed;
            // if the string parameter is not present at all, error out.
            if (mWorldRank == 0) {
               ErrorLog().printf(
                     "%s \"%s\": string parameter \"%s\" is required.\n",
                     groupKeywordFromName(groupName),
                     groupName,
                     paramName);
            }
            MPI_Barrier(mMPIComm);
            exit(EXIT_FAILURE);
         }
         else {
            *paramStringValue = nullptr;
         }
         break;
      case PARAMS_IO_WRITE: writeParamString(paramName, *paramStringValue);
   }
}

/*
 *  @groupName
 *  @paramStringName
 */
int PVParams::stringPresent(const char *groupName, const char *paramStringName) {
   ParameterGroup *g = group(groupName);
   if (g == nullptr) {
      if (mWorldRank == 0) {
         ErrorLog().printf("PVParams::stringPresent: couldn't find a group for %s\n", groupName);
      }
      exit(EXIT_FAILURE);
   }

   return g->stringPresent(paramStringName);
}

/*
 *  @groupName
 *  @paramStringName
 */
const char *
PVParams::stringValue(const char *groupName, const char *paramStringName, bool warnIfAbsent) {
   if (stringPresent(groupName, paramStringName)) {
      ParameterGroup *g = group(groupName);
      return g->stringValue(paramStringName);
   }
   else {
      if (warnIfAbsent && mWorldRank == 0) {
         WarnLog().printf(
               "No parameter string named \"%s\" in group \"%s\"\n", paramStringName, groupName);
      }
      return nullptr;
   }
}

void PVParams::writeParamString(const char *paramName, const char *svalue) {
   if (mPrintParamsStream != nullptr) {
      pvAssert(mPrintLuaStream);
      if (svalue != nullptr) {
         mPrintParamsStream->printf("    %-35s = \"%s\";\n", paramName, svalue);
         mPrintLuaStream->printf("    %-35s = \"%s\";\n", paramName, svalue);
      }
      else {
         mPrintParamsStream->printf("    %-35s = NULL;\n", paramName);
         mPrintLuaStream->printf("    %-35s = NULL;\n", paramName);
      }
   }
}

/**
 * @groupName
 */
ParameterGroup *PVParams::group(const char *groupName) {
   for (auto &g : mGroups) {
      if (std::strcmp(groupName, g->getName()) == 0) {
         return g;
      }
   }
   return nullptr;
}

const char *PVParams::groupNameFromIndex(int index) {
   bool inbounds = index >= 0 && index < getNumGroups();
   return inbounds ? mGroups[index]->getName() : nullptr;
}

const char *PVParams::groupKeywordFromIndex(int index) {
   bool inbounds = index >= 0 && index < getNumGroups();
   return inbounds ? mGroups[index]->getGroupKeyword() : nullptr;
}

const char *PVParams::groupKeywordFromName(const char *name) {
   const char *kw    = nullptr;
   ParameterGroup *g = group(name);
   if (g != nullptr) {
      kw = g->getGroupKeyword();
   }
   return kw;
}

/**
 * @keyword
 * @name
 */
void PVParams::addGroup(char *keyword, char *name) {
   // Verify that the new group's name is not an existing group's name
   for (auto &g : mGroups) {
      if (!std::strcmp(name, g->getName())) {
         Fatal().printf("Rank %d process: group name \"%s\" duplicated\n", mWorldRank, name);
      }
   }

   auto *newGroup = new ParameterGroup(name, mStack, mArrayStack, mStringStack, mWorldRank);
   mGroups.emplace_back(newGroup);
   newGroup->setGroupKeyword(keyword);

   // the parameter group takes over control of the PVParams's stack and stringStack; make new ones.
   mStack       = new ParameterStack(MAX_PARAMS);
   mArrayStack  = new ParameterArrayStack(PARAMETERARRAYSTACK_INITIALCOUNT);
   mStringStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);
}

void PVParams::addActiveParamSweep(const char *group_name, const char *param_name) {
   // Search for group_name and param_name in both ParameterSweep and BatchSweep list of objects
   for (int p = 0; p < mNumParamSweeps; p++) {
      if (std::strcmp(mParamSweeps[p]->getGroupName(), group_name) == 0
          && std::strcmp(mParamSweeps[p]->getParamName(), param_name) == 0) {
         Fatal().printf(
               "PVParams::addActiveParamSweep: Parameter sweep %s, %s already exists\n",
               group_name,
               param_name);
      }
   }

   mActiveParamSweep->setGroupAndParameter(group_name, param_name);
   ParameterSweep **newParamSweeps =
         (ParameterSweep **)calloc(mNumParamSweeps + 1, sizeof(ParameterSweep *));
   if (newParamSweeps == nullptr) {
      Fatal().printf(
            "PVParams::action_parameter_sweep: "
            "unable to allocate memory for larger ParameterSweeps\n");
   }
   for (int k = 0; k < mNumParamSweeps; k++) {
      newParamSweeps[k] = mParamSweeps[k];
   }
   free(mParamSweeps);
   mParamSweeps                  = newParamSweeps;
   mParamSweeps[mNumParamSweeps] = mActiveParamSweep;
   mNumParamSweeps++;
   newActiveParamSweep();
}

int PVParams::lookForUnread(bool errorOnUnread) {
   int status = PV_SUCCESS;
   for (int i = 0; i < getNumGroups(); i++) {
      if (mGroups[i]->lookForUnread(errorOnUnread) != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

bool PVParams::hasBeenRead(const char *group_name, const char *param_name) {
   ParameterGroup *g = group(group_name);
   if (g == nullptr) {
      return false;
   }

   return g->hasBeenRead(param_name);
}

bool PVParams::presentAndNotBeenRead(const char *group_name, const char *param_name) {
   bool is_present = present(group_name, param_name);
   if (!is_present)
      is_present = arrayPresent(group_name, param_name);
   if (!is_present)
      is_present      = stringPresent(group_name, param_name);
   bool has_been_read = hasBeenRead(group_name, param_name);
   return is_present && !has_been_read;
}

int PVParams::clearHasBeenReadFlags() {
   int status = PV_SUCCESS;
   for (int i = 0; i < getNumGroups(); i++) {
      if (mGroups[i]->clearHasBeenReadFlags() != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

void PVParams::handleUnnecessaryParameter(const char *group_name, const char *param_name) {
   if (present(group_name, param_name)) {
      if (mWorldRank == 0) {
         const char *class_name = groupKeywordFromName(group_name);
         WarnLog().printf(
               "%s \"%s\" does not use parameter %s, but it is present in the parameters file.\n",
               class_name,
               group_name,
               param_name);
      }
      value(group_name,
            param_name); // marks param as read so that presentAndNotBeenRead doesn't trip up
   }
}

void PVParams::handleUnnecessaryStringParameter(const char *group_name, const char *param_name) {
   const char *class_name = groupKeywordFromName(group_name);
   if (stringPresent(group_name, param_name)) {
      if (mWorldRank == 0) {
         WarnLog().printf(
               "%s \"%s\" does not use string parameter %s, but it is present in the parameters "
               "file.\n",
               class_name,
               group_name,
               param_name);
      }
      stringValue(group_name, param_name, false /*warnIfAbsent*/);
      // marks param as read so that presentAndNotBeenRead doesn't trip up
   }
}

void PVParams::handleUnnecessaryStringParameter(
      const char *group_name,
      const char *param_name,
      const char *correct_value,
      bool case_insensitive_flag) {
   int status             = PV_SUCCESS;
   const char *class_name = groupKeywordFromName(group_name);
   if (stringPresent(group_name, param_name)) {
      if (mWorldRank == 0) {
         WarnLog().printf(
               "%s \"%s\" does not use string parameter %s, but it is present in the parameters "
               "file.\n",
               class_name,
               group_name,
               param_name);
      }
      const char *params_value = stringValue(group_name, param_name, false /*warnIfAbsent*/);
      // marks param as read so that presentAndNotBeenRead doesn't trip up

      // Check against correct value.
      if (params_value != nullptr && correct_value != nullptr) {
         char *correct_value_i =
               strdup(correct_value); // need mutable strings for case-insensitive comparison
         char *params_value_i =
               strdup(params_value); // need mutable strings for case-insensitive comparison
         if (correct_value_i == nullptr) {
            status = PV_FAILURE;
            if (mWorldRank == 0) {
               ErrorLog().printf(
                     "%s \"%s\": Rank %d process unable to copy correct string value: %s.\n",
                     class_name,
                     group_name,
                     mWorldRank,
                     strerror(errno));
            }
         }
         if (params_value_i == nullptr) {
            status = PV_FAILURE;
            if (mWorldRank == 0) {
               ErrorLog().printf(
                     "%s \"%s\": Rank %d process unable to copy parameter string value: %s.\n",
                     class_name,
                     group_name,
                     mWorldRank,
                     strerror(errno));
            }
         }
         if (case_insensitive_flag) {
            for (char *c = params_value_i; *c != '\0'; c++) {
               *c = (char)tolower((int)*c);
            }
            for (char *c = correct_value_i; *c != '\0'; c++) {
               *c = (char)tolower((int)*c);
            }
         }
         if (std::strcmp(params_value_i, correct_value_i) != 0) {
            status = PV_FAILURE;
            if (mWorldRank == 0) {
               ErrorLog().printf(
                     "%s \"%s\": parameter string %s = \"%s\" is inconsistent with correct value "
                     "\"%s\".  Exiting.\n",
                     class_name,
                     group_name,
                     param_name,
                     params_value,
                     correct_value);
            }
         }
         free(correct_value_i);
         free(params_value_i);
      }
      else if (params_value == nullptr && correct_value != nullptr) {
         status = PV_FAILURE;
         if (mWorldRank == 0) {
            ErrorLog().printf(
                  "%s \"%s\": parameter string %s = NULL is inconsistent with correct value "
                  "\"%s\".  Exiting.\n",
                  class_name,
                  group_name,
                  param_name,
                  correct_value);
         }
      }
      else if (params_value != nullptr && correct_value == nullptr) {
         status = PV_FAILURE;
         if (mWorldRank == 0) {
            ErrorLog().printf(
                  "%s \"%s\": parameter string %s = \"%s\" is inconsistent with correct value of "
                  "NULL.  Exiting.\n",
                  class_name,
                  group_name,
                  param_name,
                  params_value);
         }
      }
      else {
         pvAssert(params_value == nullptr && correct_value == nullptr);
         pvAssert(status == PV_SUCCESS);
      }
   }
   if (status != PV_SUCCESS) {
      MPI_Barrier(mMPIComm);
      exit(EXIT_FAILURE);
   }
}

void PVParams::shuffleGroups(unsigned int seed) {
   if (seed and getNumGroups() > 1) {
      std::mt19937 shuffleRNG(seed);
      std::shuffle(mGroups.begin() + 1, mGroups.end(), shuffleRNG);
   }
}

/**
 * @id
 * @val
 */
void PVParams::action_pvparams_directive(char *id, double val) {
   if (!std::strcmp(id, "debugParsing")) {
      mDebugParsing = (val != 0);
      if (mWorldRank == 0) {
         InfoLog(directiveMessage);
         directiveMessage.printf("debugParsing turned ");
         if (mDebugParsing) {
            directiveMessage.printf("on.\n");
         }
         else {
            directiveMessage.printf("off.\n");
         }
      }
   }
   else if (!std::strcmp(id, "disable")) {
      mDisable = (val != 0);
      if (mWorldRank == 0) {
         InfoLog(directiveMessage);
         directiveMessage.printf("Parsing params file ");
         if (mDisable) {
            directiveMessage.printf("disabled.\n");
         }
         else {
            directiveMessage.printf("enabled.\n");
         }
      }
   }
   else {
      if (mWorldRank == 0) {
         WarnLog().printf("Unrecognized directive %s = %f, skipping.\n", id, val);
      }
   }
}

/**
 * @keyword
 * @name
 */
void PVParams::action_parameter_group() {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().printf(
            "action_parameter_group: %s \"%s\" parsed successfully.\n",
            mCurrGroupKeyword,
            mCurrGroupName);
      InfoLog().flush();
   }
   // build a parameter group
   addGroup(mCurrGroupKeyword, mCurrGroupName);
}
void PVParams::action_parameter_group_name(char *keyword, char *name) {
   if (mDisable)
      return;
   // remove surrounding quotes
   int len       = strlen(++name);
   name[len - 1] = '\0';

   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().printf(
            "action_parameter_group_name: %s \"%s\" parsed successfully.\n", keyword, name);
      InfoLog().flush();
   }
   mCurrGroupKeyword = keyword;
   mCurrGroupName    = name;
}

/**
 * @id
 * @val
 */
void PVParams::action_parameter_def(char *id, double val) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_def: %s = %lf\n", id, val);
      InfoLog().flush();
   }
   checkDuplicates(id);
   Parameter *p = new Parameter(id, val);
   mStack->push(p);
}

void PVParams::action_parameter_def_overwrite(char *id, double val) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_def_overwrite: %s = %lf\n", id, val);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   char *param_name     = stripOverwriteTag(id);
   Parameter *currParam = nullptr;
   for (int i = 0; i < mStack->size(); i++) {
      Parameter *param = mStack->peek(i);
      if (std::strcmp(param->getName(), param_name) == 0) {
         currParam = param;
      }
   }
   if (!currParam) {
      for (int i = 0; i < mArrayStack->size(); i++) {
         ParameterArray *arrayParam = mArrayStack->peek(i);
         if (std::strcmp(arrayParam->getName(), param_name) == 0) {
            InfoLog().flush();
            InfoLog().printf(
                  "%s is defined as an array parameter. Overwriting array parameters with value "
                  "parameters not implemented yet.\n",
                  id);
            InfoLog().flush();
         }
      }
      InfoLog().flush();
      ErrorLog().printf("Overwrite: %s is not an existing parameter to overwrite.\n", id);
      InfoLog().flush();
   }
   free(param_name);
   // Set to new value
   currParam->setValue(val);
}

void PVParams::action_parameter_array(char *id) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_array: %s\n", id);
      InfoLog().flush();
   }
   mCurrentParamArray->setName(id);
   assert(!std::strcmp(mCurrentParamArray->getName(), id));
   checkDuplicates(id);
   mArrayStack->push(mCurrentParamArray);
   mCurrentParamArray = new ParameterArray(PARAMETERARRAYSTACK_INITIALCOUNT);
}

void PVParams::action_parameter_array_overwrite(char *id) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_array_overwrite: %s\n", id);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   char *param_name          = stripOverwriteTag(id);
   ParameterArray *origArray = nullptr;
   for (int i = 0; i < mArrayStack->size(); i++) {
      ParameterArray *arrayParam = mArrayStack->peek(i);
      if (std::strcmp(arrayParam->getName(), param_name) == 0) {
         origArray = arrayParam;
      }
   }
   if (!origArray) {
      for (int i = 0; i < mStack->size(); i++) {
         Parameter *param = mStack->peek(i);
         if (std::strcmp(param->getName(), param_name) == 0) {
            InfoLog().flush();
            InfoLog().printf(
                  "%s is defined as a value parameter. Overwriting value parameters with array "
                  "parameters not implemented yet.\n",
                  id);
            InfoLog().flush();
         }
      }
      InfoLog().flush();
      ErrorLog().printf("Overwrite: %s is not an existing parameter to overwrite.\n", id);
      InfoLog().flush();
   }
   free(param_name);
   // Set values of arrays
   origArray->resetArraySize();
   for (int i = 0; i < mCurrentParamArray->getArraySize(); i++) {
      origArray->pushValue(mCurrentParamArray->peek(i));
   }
   assert(origArray->getArraySize() == mCurrentParamArray->getArraySize());
   delete mCurrentParamArray;
   mCurrentParamArray = new ParameterArray(PARAMETERARRAYSTACK_INITIALCOUNT);
}

void PVParams::action_parameter_array_value(double val) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_array_value %lf\n", val);
   }
#ifdef NDEBUG
   mCurrentParamArray->pushValue(val);
#else
   int sz            = mCurrentParamArray->getArraySize();
   int newsize       = mCurrentParamArray->pushValue(val);
   assert(newsize == sz + 1);
#endif // NDEBUG
}

void PVParams::action_parameter_string_def(const char *id, const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_string_def: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   checkDuplicates(id);
   char *param_value = stripQuotationMarks(stringval);
   assert(!stringval || param_value); // stringval can be null, but if stringval is not null,
   // param_value should also be non-null
   ParameterString *pstr = new ParameterString(id, param_value);
   mStringStack->push(pstr);
   free(param_value);
}

void PVParams::action_parameter_string_def_overwrite(const char *id, const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_string_def_overwrite: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   char *param_name           = stripOverwriteTag(id);
   ParameterString *currParam = nullptr;
   for (int i = 0; i < mStringStack->size(); i++) {
      ParameterString *param = mStringStack->peek(i);
      assert(param);
      if (std::strcmp(param->getName(), param_name) == 0) {
         currParam = param;
      }
   }
   free(param_name);
   if (!currParam) {
      ErrorLog().printf("Overwrite: %s is not an existing parameter to overwrite.\n", id);
   }
   char *param_value = stripQuotationMarks(stringval);
   assert(!stringval || param_value); // stringval can be null, but if stringval is not null,
   // param_value should also be non-null
   // Set to new value
   currParam->setValue(param_value);
   free(param_value);
}

void PVParams::action_parameter_filename_def(const char *id, const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_filename_def: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   checkDuplicates(id);
   char *param_value = stripQuotationMarks(stringval);
   assert(param_value);
   ParameterString *pstr = new ParameterString(id, param_value);
   free(param_value);
   mStringStack->push(pstr);
}

void PVParams::action_parameter_filename_def_overwrite(const char *id, const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_filename_def_overwrite: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   char *param_name           = stripOverwriteTag(id);
   ParameterString *currParam = nullptr;
   for (int i = 0; i < mStringStack->size(); i++) {
      ParameterString *param = mStringStack->peek(i);
      assert(param);
      if (std::strcmp(param->getName(), param_name) == 0) {
         currParam = param;
      }
   }
   free(param_name);
   param_name = nullptr;
   if (!currParam) {
      ErrorLog().printf("Overwrite: %s is not an existing parameter to overwrite.\n", id);
   }
   char *param_value = stripQuotationMarks(stringval);
   assert(param_value);
   currParam->setValue(param_value);
   free(param_value);
}

void PVParams::action_include_directive(const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_include_directive: including %s\n", stringval);
      InfoLog().flush();
   }
   // The include directive must be the first parameter in the group if defined
   if (mStack->size() != 0 || mArrayStack->size() != 0 || mStringStack->size() != 0) {
      ErrorLog().printf(
            "Import of %s must be the first parameter specified in the group.\n", stringval);
      InfoLog().flush();
   }
   // Grab the parameter value
   char *param_value = stripQuotationMarks(stringval);
   // Grab the included group's ParameterGroup object
   ParameterGroup *includeGroup = nullptr;
   for (auto &g : mGroups) {
      // If strings are matching
      if (std::strcmp(g->getName(), param_value) == 0) {
         includeGroup = g;
      }
   }
   // If group not found
   if (!includeGroup) {
      ErrorLog().printf("Include: include group %s is not defined.\n", param_value);
   }
   // Check keyword of group
   if (std::strcmp(includeGroup->getGroupKeyword(), mCurrGroupKeyword) != 0) {
      ErrorLog().printf(
            "Include: Cannot include group %s, which is a %s, into a %s. Group types must be the "
            "same.\n",
            param_value,
            includeGroup->getGroupKeyword(),
            mCurrGroupKeyword);
   }
   free(param_value);
   // Load all stack values into current parameter group

   assert(mStack->size() == 0);
   delete mStack;
   mStack = includeGroup->copyStack();

   assert(mArrayStack->size() == 0);
   delete mArrayStack;
   mArrayStack = includeGroup->copyArrayStack();

   assert(mStringStack->size() == 0);
   delete mStringStack;
   mStringStack = includeGroup->copyStringStack();
}

void PVParams::action_parameter_sweep_open(const char *groupname, const char *paramname) {
   if (mDisable)
      return;
   // strip quotation marks from groupname
   mCurrSweepGroupName = stripQuotationMarks(groupname);
   assert(mCurrSweepGroupName);
   mCurrSweepParamName = strdup(paramname);
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf(
            "action_parameter_sweep_open: Sweep for group %s, parameter \"%s\" starting\n",
            groupname,
            paramname);
      InfoLog().flush();
   }
}

void PVParams::action_parameter_sweep_close() {
   if (mDisable)
      return;
   addActiveParamSweep(mCurrSweepGroupName, mCurrSweepParamName);
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().printf(
            "action_parameter_group: ParameterSweep for %s \"%s\" parsed successfully.\n",
            mCurrSweepGroupName,
            mCurrSweepParamName);
      InfoLog().flush();
   }
   // build a parameter group
   free(mCurrSweepGroupName);
   free(mCurrSweepParamName);
}

void PVParams::action_parameter_sweep_values_number(double val) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_sweep_values_number: %f\n", val);
      InfoLog().flush();
   }
   mActiveParamSweep->pushNumericValue(val);
}

void PVParams::action_parameter_sweep_values_string(const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_sweep_values_string: %s\n", stringval);
      InfoLog().flush();
   }
   char *string = stripQuotationMarks(stringval);
   assert(!stringval || string); // stringval can be null, but if stringval is not null, string
   // should also be non-null
   mActiveParamSweep->pushStringValue(string);
   free(string);
}

void PVParams::action_parameter_sweep_values_filename(const char *stringval) {
   if (mDisable)
      return;
   if (mDebugParsing && mWorldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_sweep_values_filename: %s\n", stringval);
      InfoLog().flush();
   }
   char *filename = stripQuotationMarks(stringval);
   assert(filename);
   mActiveParamSweep->pushStringValue(filename);
   free(filename);
}

void PVParams::checkDuplicates(const char *paramName) {
   bool hasDuplicate = false;
   for (int k = 0; k < mStack->size(); k++) {
      Parameter *parm = mStack->peek(k);
      if (!std::strcmp(paramName, parm->getName())) {
         ErrorLog().printf(
               "Rank %d process: The params group for %s \"%s\" duplicates "
               "parameter \"%s\".\n",
               mWorldRank,
               mCurrGroupKeyword,
               mCurrGroupName,
               paramName);
         hasDuplicate = true;
      }
   }
   for (int k = 0; k < mArrayStack->size(); k++) {
      if (!std::strcmp(paramName, mArrayStack->peek(k)->getName())) {
         ErrorLog().printf(
               "Rank %d process: The params group for %s \"%s\" duplicates "
               "array parameter \"%s\".\n",
               mWorldRank,
               mCurrGroupKeyword,
               mCurrGroupName,
               paramName);
         hasDuplicate = true;
      }
   }
   for (int k = 0; k < mStringStack->size(); k++) {
      if (!std::strcmp(paramName, mStringStack->peek(k)->getName())) {
         ErrorLog().printf(
               "Rank %d process: The params group for %s \"%s\" duplicates "
               "string parameter \"%s\".\n",
               mWorldRank,
               mCurrGroupKeyword,
               mCurrGroupName,
               paramName);
         hasDuplicate = true;
      }
   }
   if (hasDuplicate) {
      exit(EXIT_FAILURE);
   }
}

char *PVParams::stripQuotationMarks(const char *s) {
   // If a string has quotes as its first and last character, return the
   // part of the string inside the quotes, e.g. {'"', 'c', 'a', 't', '"'}
   // becomes {'c', 'a', 't'}.  If the string is null or does not have quotes at the
   // beginning and end, return nullptr.
   // It is the responsibility of the routine that calls stripQuotationMarks
   // to free the returned string to avoid a memory leak.
   if (s == nullptr) {
      return nullptr;
   }
   char *noquotes = nullptr;
   int len        = strlen(s);
   if (len >= 2 && s[0] == '"' && s[len - 1] == '"') {
      noquotes = (char *)calloc(len - 1, sizeof(char));
      memcpy(noquotes, s + 1, len - 2);
      noquotes[len - 2] = '\0'; // Not strictly necessary since noquotes was calloc'ed
   }
   return noquotes;
}

char *PVParams::stripOverwriteTag(const char *s) {
   // Strips the @ tag to any overwritten params
   int len     = strlen(s);
   char *notag = nullptr;
   if (len >= 1 && s[0] == '@') {
      notag = (char *)calloc(len, sizeof(char));
      memcpy(notag, s + 1, len - 1);
      notag[len - 1] = '\0';
   }
   return notag;
}

} // close namespace PV block
