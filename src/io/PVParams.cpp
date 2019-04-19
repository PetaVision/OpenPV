/*
 * PVParams.cpp
 *
 *  Created on: Nov 27, 2008
 *      Author: rasmussn
 */

#include "PVParams.hpp"
#include "include/pv_common.h"
#include "utils/PVAlloc.hpp"
#include <assert.h>
#include <climits> // INT_MIN
#include <cmath> // nearbyint()
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
   paramName       = strdup(name);
   paramValue      = (float)value;
   paramDblValue   = value;
   hasBeenReadFlag = false;
}

Parameter::~Parameter() { free(paramName); }

ParameterArray::ParameterArray(int initialSize) {
   hasBeenReadFlag = false;
   paramNameSet    = false;
   paramName       = strdup("Unnamed Parameter array");
   bufferSize      = initialSize;
   arraySize       = 0;
   values          = NULL;
   if (bufferSize > 0) {
      values    = (float *)calloc(bufferSize, sizeof(float));
      valuesDbl = (double *)calloc(bufferSize, sizeof(double));
      if (values == NULL || valuesDbl == NULL) {
         Fatal().printf("ParameterArray failed to allocate memory for \"%s\"\n", name());
      }
   }
}

ParameterArray::~ParameterArray() {
   free(paramName);
   paramName = NULL;
   free(values);
   values = NULL;
   free(valuesDbl);
   valuesDbl = NULL;
}

int ParameterArray::setName(const char *name) {
   int status = PV_SUCCESS;
   if (paramNameSet == false) {
      free(paramName);
      paramName    = strdup(name);
      paramNameSet = true;
   }
   else {
      ErrorLog().printf(
            "ParameterArray::setName called with \"%s\" but name is already set to \"%s\"\n",
            name,
            paramName);
      status = PV_FAILURE;
   }
   return status;
}

int ParameterArray::pushValue(double value) {
   assert(bufferSize >= arraySize);
   if (bufferSize == arraySize) {
      bufferSize += PARAMETERARRAY_INITIALSIZE;
      float *new_values = (float *)calloc(bufferSize, sizeof(float));
      if (new_values == NULL) {
         Fatal().printf(
               "ParameterArray::pushValue failed to increase array \"%s\" to %d values\n",
               name(),
               arraySize + 1);
      }
      memcpy(new_values, values, sizeof(float) * arraySize);
      free(values);
      values                 = new_values;
      double *new_values_dbl = (double *)calloc(bufferSize, sizeof(double));
      if (new_values == NULL) {
         Fatal().printf(
               "ParameterArray::pushValue failed to increase array \"%s\" to %d values\n",
               name(),
               arraySize + 1);
      }
      memcpy(new_values_dbl, valuesDbl, sizeof(double) * arraySize);
      free(valuesDbl);
      valuesDbl = new_values_dbl;
   }
   assert(arraySize < bufferSize);
   valuesDbl[arraySize] = value;
   values[arraySize]    = (float)value;
   arraySize++;
   return arraySize;
}

ParameterArray *ParameterArray::copyParameterArray() {
   ParameterArray *returnPA = new ParameterArray(bufferSize);
   returnPA->setName(paramName);
   assert(!strcmp(returnPA->name(), paramName));
   for (int i = 0; i < arraySize; i++) {
      returnPA->pushValue(valuesDbl[i]);
   }
   return returnPA;
}

/**
 * @name
 * @value
 */
ParameterString::ParameterString(const char *name, const char *value) {
   paramName       = name ? strdup(name) : NULL;
   paramValue      = value ? strdup(value) : NULL;
   hasBeenReadFlag = false;
}

ParameterString::~ParameterString() {
   free(paramName);
   free(paramValue);
}

/**
 * @maxCount
 */
ParameterStack::ParameterStack(int maxCount) {
   this->maxCount = maxCount;
   count          = 0;
   parameters     = (Parameter **)malloc(maxCount * sizeof(Parameter *));
}

ParameterStack::~ParameterStack() {
   for (int i = 0; i < count; i++) {
      delete parameters[i];
   }
   free(parameters);
}

/**
 * @param
 */
int ParameterStack::push(Parameter *param) {
   assert(count < maxCount);
   parameters[count++] = param;
   return 0;
}

Parameter *ParameterStack::pop() {
   assert(count > 0);
   return parameters[count--];
}

ParameterArrayStack::ParameterArrayStack(int initialCount) {
   allocation      = initialCount;
   count           = 0;
   parameterArrays = NULL;
   if (initialCount > 0) {
      parameterArrays = (ParameterArray **)calloc(allocation, sizeof(ParameterArray *));
      if (parameterArrays == NULL) {
         Fatal().printf(
               "ParameterArrayStack unable to allocate %d parameter arrays\n", initialCount);
      }
   }
}

ParameterArrayStack::~ParameterArrayStack() {
   for (int k = 0; k < count; k++) {
      delete parameterArrays[k];
      parameterArrays[k] = NULL;
   }
   free(parameterArrays);
   parameterArrays = NULL;
}

int ParameterArrayStack::push(ParameterArray *array) {
   assert(count <= allocation);
   if (count == allocation) {
      int newallocation = allocation + RESIZE_ARRAY_INCR;
      ParameterArray **newParameterArrays =
            (ParameterArray **)malloc(newallocation * sizeof(ParameterArray *));
      if (!newParameterArrays)
         return PV_FAILURE;
      for (int i = 0; i < count; i++) {
         newParameterArrays[i] = parameterArrays[i];
      }
      allocation = newallocation;
      free(parameterArrays);
      parameterArrays = newParameterArrays;
   }
   assert(count < allocation);
   parameterArrays[count] = array;
   count++;
   return PV_SUCCESS;
}

/*
 * initialCount
 */
ParameterStringStack::ParameterStringStack(int initialCount) {
   allocation       = initialCount;
   count            = 0;
   parameterStrings = (ParameterString **)calloc(allocation, sizeof(ParameterString *));
}

ParameterStringStack::~ParameterStringStack() {
   for (int i = 0; i < count; i++) {
      delete parameterStrings[i];
   }
   free(parameterStrings);
}

/*
 * @param
 */
int ParameterStringStack::push(ParameterString *param) {
   assert(count <= allocation);
   if (count == allocation) {
      int newallocation = allocation + RESIZE_ARRAY_INCR;
      ParameterString **newparameterStrings =
            (ParameterString **)malloc(newallocation * sizeof(ParameterString *));
      if (!newparameterStrings)
         return PV_FAILURE;
      for (int i = 0; i < count; i++) {
         newparameterStrings[i] = parameterStrings[i];
      }
      allocation = newallocation;
      free(parameterStrings);
      parameterStrings = newparameterStrings;
   }
   assert(count < allocation);
   parameterStrings[count++] = param;
   return PV_SUCCESS;
}

ParameterString *ParameterStringStack::pop() {
   if (count > 0) {
      return parameterStrings[count--];
   }
   else
      return NULL;
}

const char *ParameterStringStack::lookup(const char *targetname) {
   const char *result = NULL;
   for (int i = 0; i < count; i++) {
      if (!strcmp(parameterStrings[i]->getName(), targetname)) {
         result = parameterStrings[i]->getValue();
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
   this->groupName    = strdup(name);
   this->groupKeyword = NULL;
   this->stack        = stack;
   this->arrayStack   = array_stack;
   this->stringStack  = string_stack;
   this->processRank  = rank;
}

ParameterGroup::~ParameterGroup() {
   free(groupName);
   groupName = NULL;
   free(groupKeyword);
   groupKeyword = NULL;
   delete stack;
   stack = NULL;
   delete arrayStack;
   arrayStack = NULL;
   delete stringStack;
   stringStack = NULL;
}

int ParameterGroup::setGroupKeyword(const char *keyword) {
   if (groupKeyword == NULL) {
      size_t keywordlen = strlen(keyword);
      groupKeyword      = (char *)malloc(keywordlen + 1);
      if (groupKeyword) {
         strcpy(groupKeyword, keyword);
      }
   }
   return groupKeyword == NULL ? PV_FAILURE : PV_SUCCESS;
}

int ParameterGroup::setStringStack(ParameterStringStack *stringStack) {
   this->stringStack = stringStack;
   // ParameterGroup::setStringStack takes ownership of the stringStack;
   // i.e. it will delete it when the ParameterGroup is deleted.
   // You shouldn't use a stringStack after calling this routine with it.
   // Instead, query it with ParameterGroup::stringPresent and
   // ParameterGroup::stringValue methods.
   return stringStack == NULL ? PV_FAILURE : PV_SUCCESS;
}

/**
 * @name
 */
int ParameterGroup::present(const char *name) {
   int count = stack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = stack->peek(i);
      if (strcmp(name, p->name()) == 0) {
         return 1; // string is present
      }
   }
   return 0; // string not present
}

/**
 * @name
 */
double ParameterGroup::value(const char *name) {
   int count = stack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = stack->peek(i);
      if (strcmp(name, p->name()) == 0) {
         return p->value();
      }
   }
   Fatal().printf(
         "PVParams::ParameterGroup::value: ERROR, couldn't find a value for %s"
         " in group %s\n",
         name,
         groupName);
   return PV_FAILURE; // suppresses warning in compilers that don't recognize Fatal always exits.
}

bool ParameterGroup::arrayPresent(const char *name) {
   bool array_found = false;
   int count        = arrayStack->size();
   for (int i = 0; i < count; i++) {
      ParameterArray *p = arrayStack->peek(i);
      if (strcmp(name, p->name()) == 0) {
         array_found = true; // string is present
         break;
      }
   }
   if (!array_found) {
      array_found = (present(name) != 0);
   }
   return array_found;
}

const float *ParameterGroup::arrayValues(const char *name, int *size) {
   int count         = arrayStack->size();
   *size             = 0;
   const float *v    = NULL;
   ParameterArray *p = NULL;
   for (int i = 0; i < count; i++) {
      p = arrayStack->peek(i);
      if (strcmp(name, p->name()) == 0) {
         v = p->getValues(size);
         break;
      }
   }
   if (!v) {
      Parameter *q = NULL;
      for (int i = 0; i < stack->size(); i++) {
         Parameter *q1 = stack->peek(i);
         assert(q1);
         if (strcmp(name, q1->name()) == 0) {
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
   int count         = arrayStack->size();
   *size             = 0;
   const double *v   = NULL;
   ParameterArray *p = NULL;
   for (int i = 0; i < count; i++) {
      p = arrayStack->peek(i);
      if (strcmp(name, p->name()) == 0) {
         v = p->getValuesDbl(size);
         break;
      }
   }
   if (!v) {
      Parameter *q = NULL;
      for (int i = 0; i < stack->size(); i++) {
         Parameter *q1 = stack->peek(i);
         assert(q1);
         if (strcmp(name, q1->name()) == 0) {
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
   // not really necessary, as stringValue returns NULL if the
   // string is not found, but included on the analogy with
   // value and present methods for floating-point parameters
   if (!stringName)
      return 0;
   int count = stringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *pstr = stringStack->peek(i);
      assert(pstr);
      if (!strcmp(stringName, pstr->getName())) {
         return 1; // string is present
      }
   }
   return 0; // string not present
}

const char *ParameterGroup::stringValue(const char *stringName) {
   if (!stringName)
      return NULL;
   int count = stringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *pstr = stringStack->peek(i);
      assert(pstr);
      if (!strcmp(stringName, pstr->getName())) {
         return pstr->getValue();
      }
   }
   return NULL;
}

int ParameterGroup::warnUnread() {
   int status = PV_SUCCESS;
   int count;
   count = stack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = stack->peek(i);
      if (!p->hasBeenRead()) {
         if (processRank == 0)
            WarnLog().printf(
                  "Parameter group \"%s\": parameter \"%s\" has not been read.\n",
                  name(),
                  p->name());
         status = PV_FAILURE;
      }
   }
   count = arrayStack->size();
   for (int i = 0; i < count; i++) {
      ParameterArray *parr = arrayStack->peek(i);
      if (!parr->hasBeenRead()) {
         if (processRank == 0)
            WarnLog().printf(
                  "Parameter group \"%s\": array parameter \"%s\" has not been read.\n",
                  name(),
                  parr->name());
      }
   }
   count = stringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *pstr = stringStack->peek(i);
      if (!pstr->hasBeenRead()) {
         if (processRank == 0)
            WarnLog().printf(
                  "Parameter group \"%s\": string parameter \"%s\" has not been read.\n",
                  name(),
                  pstr->getName());
         status = PV_FAILURE;
      }
   }
   return status;
}

bool ParameterGroup::hasBeenRead(const char *paramName) {
   int count;
   count = stack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = stack->peek(i);
      if (!strcmp(p->name(), paramName)) {
         return p->hasBeenRead();
      }
   }
   count = arrayStack->size();
   for (int i = 0; i < count; i++) {
      ParameterArray *parr = arrayStack->peek(i);
      if (!strcmp(parr->name(), paramName)) {
         return parr->hasBeenRead();
      }
   }
   count = stringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *pstr = stringStack->peek(i);
      if (!strcmp(pstr->getName(), paramName)) {
         return pstr->hasBeenRead();
      }
   }
   return false;
}

int ParameterGroup::clearHasBeenReadFlags() {
   int status = PV_SUCCESS;
   int count  = stack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = stack->peek(i);
      p->clearHasBeenRead();
   }
   count = arrayStack->size();
   for (int i = 0; i < count; i++) {
      ParameterArray *parr = arrayStack->peek(i);
      parr->clearHasBeenRead();
   }
   count = stringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *pstr = stringStack->peek(i);
      pstr->clearHasBeenRead();
   }
   return status;
}

int ParameterGroup::pushNumerical(Parameter *param) { return stack->push(param); }

int ParameterGroup::pushString(ParameterString *param) { return stringStack->push(param); }

int ParameterGroup::setValue(const char *param_name, double value) {
   int status = PV_SUCCESS;
   int count  = stack->size();
   for (int i = 0; i < count; i++) {
      Parameter *p = stack->peek(i);
      if (strcmp(param_name, p->name()) == 0) {
         p->setValue(value);
         return PV_SUCCESS;
      }
   }
   Fatal().printf(
         "PVParams::ParameterGroup::setValue: ERROR, couldn't find parameter %s"
         " in group \"%s\"\n",
         param_name,
         name());

   return status;
}

int ParameterGroup::setStringValue(const char *param_name, const char *svalue) {
   int status = PV_SUCCESS;
   int count  = stringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString *p = stringStack->peek(i);
      if (strcmp(param_name, p->getName()) == 0) {
         p->setValue(svalue);
         return PV_SUCCESS;
      }
   }
   Fatal().printf(
         "PVParams::ParameterGroup::setStringValue: ERROR, couldn't find a string value for %s"
         " in group \"%s\"\n",
         param_name,
         name());

   return status;
}

/**
 * A function to return a copy of the parameter group's stack.
 */
ParameterStack *ParameterGroup::copyStack() {
   ParameterStack *returnStack = new ParameterStack(MAX_PARAMS);
   for (int i = 0; i < stack->size(); i++) {
      returnStack->push(stack->peek(i)->copyParameter());
   }
   return returnStack;
}

/**
 * A function to return a copy of the parameter group's stack.
 */
ParameterArrayStack *ParameterGroup::copyArrayStack() {
   ParameterArrayStack *returnStack = new ParameterArrayStack(PARAMETERARRAYSTACK_INITIALCOUNT);
   for (int i = 0; i < arrayStack->size(); i++) {
      returnStack->push(arrayStack->peek(i)->copyParameterArray());
   }
   return returnStack;
}

/**
 * A function to return a copy of the parameter group's stack.
 */
ParameterStringStack *ParameterGroup::copyStringStack() {
   ParameterStringStack *returnStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);
   for (int i = 0; i < stringStack->size(); i++) {
      returnStack->push(stringStack->peek(i)->copyParameterString());
   }
   return returnStack;
}

ParameterSweep::ParameterSweep() {
   groupName         = NULL;
   paramName         = NULL;
   numValues         = 0;
   currentBufferSize = 0;
   type              = SWEEP_UNDEF;
   valuesNumber      = NULL;
   valuesString      = NULL;
}

ParameterSweep::~ParameterSweep() {
   free(groupName);
   groupName = NULL;
   free(paramName);
   paramName = NULL;
   free(valuesNumber);
   valuesNumber = NULL;
   if (valuesString != NULL) {
      for (int k = 0; k < numValues; k++) {
         free(valuesString[k]);
      }
      free(valuesString);
      valuesString = NULL;
   }
}

int ParameterSweep::setGroupAndParameter(const char *groupname, const char *parametername) {
   int status = PV_SUCCESS;
   if (groupName != NULL || paramName != NULL) {
      ErrorLog(errorMessage);
      errorMessage.printf("ParameterSweep::setGroupParameter: ");
      if (groupName != NULL) {
         errorMessage.printf(" groupName has already been set to \"%s\".", groupName);
      }
      if (paramName != NULL) {
         errorMessage.printf(" paramName has already been set to \"%s\".", paramName);
      }
      errorMessage.printf("\n");
      status = PV_FAILURE;
   }
   else {
      groupName = strdup(groupname);
      paramName = strdup(parametername);
      // Check for duplicates
   }
   return status;
}

int ParameterSweep::pushNumericValue(double val) {
   int status = PV_SUCCESS;
   if (numValues == 0) {
      type = SWEEP_NUMBER;
   }
   assert(type == SWEEP_NUMBER);
   assert(valuesString == NULL);

   assert(numValues <= currentBufferSize);
   if (numValues == currentBufferSize) {
      currentBufferSize += PARAMETERSWEEP_INCREMENTCOUNT;
      double *newValuesNumber = (double *)calloc(currentBufferSize, sizeof(double));
      if (newValuesNumber == NULL) {
         ErrorLog().printf("ParameterSweep:pushNumericValue: unable to allocate memory\n");
         status = PV_FAILURE;
         abort();
      }
      for (int k = 0; k < numValues; k++) {
         newValuesNumber[k] = valuesNumber[k];
      }
      free(valuesNumber);
      valuesNumber = newValuesNumber;
   }
   valuesNumber[numValues] = val;
   numValues++;
   return status;
}

int ParameterSweep::pushStringValue(const char *sval) {
   int status = PV_SUCCESS;
   if (numValues == 0) {
      type = SWEEP_STRING;
   }
   assert(type == SWEEP_STRING);
   assert(valuesNumber == NULL);

   assert(numValues <= currentBufferSize);
   if (numValues == currentBufferSize) {
      currentBufferSize += PARAMETERSWEEP_INCREMENTCOUNT;
      char **newValuesString = (char **)calloc(currentBufferSize, sizeof(char *));
      if (newValuesString == NULL) {
         ErrorLog().printf("ParameterSweep:pushStringValue: unable to allocate memory\n");
         status = PV_FAILURE;
         abort();
      }
      for (int k = 0; k < numValues; k++) {
         newValuesString[k] = valuesString[k];
      }
      free(valuesString);
      valuesString = newValuesString;
   }
   valuesString[numValues] = strdup(sval);
   numValues++;
   return status;
}

int ParameterSweep::getNumericValue(int n, double *val) {
   int status = PV_SUCCESS;
   assert(valuesNumber != NULL);
   if (type != SWEEP_NUMBER || n < 0 || n >= numValues) {
      status = PV_FAILURE;
   }
   else {
      *val = valuesNumber[n];
   }
   return status;
}

const char *ParameterSweep::getStringValue(int n) {
   char *str = NULL;
   assert(valuesString != NULL);
   if (type == SWEEP_STRING && n >= 0 && n < numValues) {
      str = valuesString[n];
   }
   return str;
}

/**
 * @filename
 * @initialSize
 * @icComm
 */
PVParams::PVParams(const char *filename, size_t initialSize, Communicator const *inIcComm) {
   this->icComm = inIcComm;
   initialize(initialSize);
   parseFile(filename);
}

/*
 * @initialSize
 * @icComm
 */
PVParams::PVParams(size_t initialSize, Communicator const *inIcComm) {
   this->icComm = inIcComm;
   initialize(initialSize);
}

/*
 * @buffer
 * @bufferLength
 * @initialSize
 * @icComm
 */
PVParams::PVParams(
      const char *buffer,
      long int bufferLength,
      size_t initialSize,
      Communicator const *inIcComm) {
   this->icComm = inIcComm;
   initialize(initialSize);
   parseBuffer(buffer, bufferLength);
}

PVParams::~PVParams() {
   for (int i = 0; i < numGroups; i++) {
      delete groups[i];
   }
   free(groups);
   delete currentParamArray;
   currentParamArray = NULL;
   delete stack;
   delete arrayStack;
   delete stringStack;
   delete this->activeParamSweep;
   for (int i = 0; i < numParamSweeps; i++) {
      delete paramSweeps[i];
   }
   free(paramSweeps);
   paramSweeps = NULL;
}

/*
 * @initialSize
 */
int PVParams::initialize(size_t initialSize) {
   this->numGroups = 0;
   groupArraySize  = initialSize;
   // Get world rank and size
   MPI_Comm_rank(icComm->globalCommunicator(), &worldRank);
   MPI_Comm_size(icComm->globalCommunicator(), &worldSize);

   groups      = (ParameterGroup **)malloc(initialSize * sizeof(ParameterGroup *));
   stack       = new ParameterStack(MAX_PARAMS);
   arrayStack  = new ParameterArrayStack(PARAMETERARRAYSTACK_INITIALCOUNT);
   stringStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);

   currentParamArray = new ParameterArray(PARAMETERARRAYSTACK_INITIALCOUNT);

   numParamSweeps = 0;
   paramSweeps    = NULL;
   newActiveParamSweep();
#ifdef DEBUG_PARSING
   debugParsing = true;
#else
   debugParsing      = false;
#endif // DEBUG_PARSING
   disable = false;

   return (groups && stack && stringStack && activeParamSweep) ? PV_SUCCESS : PV_FAILURE;
}

int PVParams::newActiveParamSweep() {
   int status       = PV_SUCCESS;
   activeParamSweep = new ParameterSweep();
   if (activeParamSweep == NULL) {
      Fatal().printf("PVParams::newActiveParamSweep: unable to create activeParamSweep");
      status = PV_FAILURE;
   }
   return status;
}

int PVParams::parseFile(const char *filename) {
   int rootproc      = 0;
   char *paramBuffer = nullptr;
   size_t bufferlen;
   if (worldRank == rootproc) {
      std::string paramBufferString("");
      loadParamBuffer(filename, paramBufferString);
      bufferlen = paramBufferString.size();
      // Older versions of MPI_Send require void*, not void const*
      paramBuffer = (char *)pvMalloc(bufferlen + 1);
      memcpy(paramBuffer, paramBufferString.c_str(), bufferlen);
      paramBuffer[bufferlen] = '\0';

#ifdef PV_USE_MPI
      int sz = worldSize;
      for (int i = 0; i < sz; i++) {
         if (i == rootproc)
            continue;
         MPI_Send(paramBuffer, (int)bufferlen, MPI_CHAR, i, 31, icComm->globalCommunicator());
      }
#endif // PV_USE_MPI
   }
   else { // rank != rootproc
#ifdef PV_USE_MPI
      MPI_Status mpi_status;
      int count;
      MPI_Probe(rootproc, 31, icComm->globalCommunicator(), &mpi_status);
      // int status =
      MPI_Get_count(&mpi_status, MPI_CHAR, &count);
      bufferlen   = (size_t)count;
      paramBuffer = (char *)malloc(bufferlen);
      if (paramBuffer == NULL) {
         Fatal().printf(
               "PVParams::parseFile: Rank %d process unable to allocate memory for params buffer\n",
               worldRank);
      }
      MPI_Recv(
            paramBuffer,
            (int)bufferlen,
            MPI_CHAR,
            rootproc,
            31,
            icComm->globalCommunicator(),
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
   bool const useLua        = fnlen >= luaextlen && !strcmp(&filename[fnlen - luaextlen], luaext);
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
   for (int k = 0; k < numberOfParameterSweeps(); k++) {
      ParameterSweep *sweep  = paramSweeps[k];
      group_name             = sweep->getGroupName();
      const char *param_name = sweep->getParamName();
      ParameterGroup *gp     = group(group_name);
      if (gp == NULL) {
         Fatal().printf(
               "PVParams::hasSweepValue error: ParameterSweep %d (zero-indexed) refers to "
               "non-existent group \"%s\"\n",
               k,
               group_name);
      }
      if (!strcmp(gp->getGroupKeyword(), "HyPerCol") && !strcmp(param_name, inParamName)) {
         out = true;
         break;
      }
   }
   return out;
}

int PVParams::parseBuffer(char const *buffer, long int bufferLength) {
   // Assumes that each MPI process has the same contents in buffer.

   // This is where it calls the scanner and parser
   parseStatus = pv_parseParameters(this, buffer, bufferLength);
   if (parseStatus != 0) {
      ErrorLog().printf(
            "Rank %d process: pv_parseParameters failed with return value %d\n",
            worldRank,
            parseStatus);
   }
   getOutputStream().flush();

   setParameterSweepSize(); // Need to set sweepSize here, because if the outputPath sweep needs to
   // be created we need to know the size.

   // If there is at least one ParameterSweep  and none of them set outputPath, create a
   // parameterSweep that does set outputPath.

   // If both parameterSweep and batchSweep is set, must autoset output path, as there is no way to
   // specify both paramSweep and batchSweep
   if (numberOfParameterSweeps() > 0) {
      if (!hasSweepValue("outputPath")) {
         const char *hypercolgroupname = NULL;
         const char *outputPathName    = NULL;
         for (int g = 0; g < numGroups; g++) {
            if (groups[g]->getGroupKeyword(), "HyPerCol") {
               hypercolgroupname = groups[g]->name();
               outputPathName    = groups[g]->stringValue("outputPath");
               if (outputPathName == NULL) {
                  Fatal().printf(
                        "PVParams::outputPath must be specified if parameterSweep does "
                        "not sweep over outputPath\n");
               }
               break;
            }
         }
         if (hypercolgroupname == NULL) {
            ErrorLog().printf("PVParams::parseBuffer: no HyPerCol group\n");
            abort();
         }

         // Push the strings "[outputPathName]/paramsweep_[n]/"
         // to the parameter sweep, where [n] ranges from 0 to parameterSweepSize - 1,
         // and is zero-padded so that the parameter sweep's outputPath directories
         // sort the same lexicographically and numerically.
         auto lenmax = std::to_string(parameterSweepSize - 1).size();
         for (int i = 0; i < parameterSweepSize; i++) {
            std::string outputPathStr(outputPathName);
            outputPathStr.append("/paramsweep_");
            std::string serialNumberStr = std::to_string(i);
            auto len                    = serialNumberStr.size();
            if (len < lenmax) {
               outputPathStr.append(lenmax - len, '0');
            }
            outputPathStr.append(serialNumberStr);
            outputPathStr.append("/");
            activeParamSweep->pushStringValue(outputPathStr.c_str());
         }
         addActiveParamSweep(hypercolgroupname, "outputPath");
      }

      if (!hasSweepValue("checkpointWriteDir")) {
         const char *hypercolgroupname  = NULL;
         const char *checkpointWriteDir = NULL;
         for (int g = 0; g < numGroups; g++) {
            if (groups[g]->getGroupKeyword(), "HyPerCol") {
               hypercolgroupname  = groups[g]->name();
               checkpointWriteDir = groups[g]->stringValue("checkpointWriteDir");
               // checkpointWriteDir can be NULL if checkpointWrite is set to false
               break;
            }
         }
         if (hypercolgroupname == NULL) {
            ErrorLog().printf("PVParams::parseBuffer: no HyPerCol group\n");
            abort();
         }
         if (checkpointWriteDir) {
            // Push the strings "[checkpointWriteDir]/paramsweep_[n]/"
            // to the parameter sweep, where [n] ranges from 0 to parameterSweepSize - 1,
            // and is zero-padded so that the parameter sweep's checkpointWriteDir directories
            // sort the same lexicographically and numerically.
            auto lenmax = std::to_string(parameterSweepSize - 1).size();
            for (int i = 0; i < parameterSweepSize; i++) {
               std::string checkpointWriteDirStr(checkpointWriteDir);
               checkpointWriteDirStr.append("/paramsweep_");
               std::string serialNumberStr = std::to_string(i);
               auto len                    = serialNumberStr.size();
               if (len < lenmax) {
                  checkpointWriteDirStr.append(lenmax - len, '0');
               }
               checkpointWriteDirStr.append(serialNumberStr);
               checkpointWriteDirStr.append("/");
               activeParamSweep->pushStringValue(checkpointWriteDirStr.c_str());
            }
            addActiveParamSweep(hypercolgroupname, "checkpointWriteDir");
         }
      }
   }

   if (icComm->numCommBatches() > 1) {
      ParameterGroup *hypercolGroup = nullptr;
      for (int g = 0; g < numGroups; g++) {
         ParameterGroup *group = groups[g];
         if (!strcmp(group->getGroupKeyword(), "HyPerCol")) {
            hypercolGroup = group;
            break;
         }
      }
      FatalIf(hypercolGroup == nullptr, "PVParams::parseBuffer: no HyPerCol group\n");
   }

   // Each ParameterSweep needs to have its group/parameter pair added to the database, if it's not
   // already present.
   for (int k = 0; k < numberOfParameterSweeps(); k++) {
      ParameterSweep *sweep  = paramSweeps[k];
      const char *group_name = sweep->getGroupName();
      const char *param_name = sweep->getParamName();
      SweepType type         = sweep->getType();
      ParameterGroup *g      = group(group_name);
      if (g == NULL) {
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
   parameterSweepSize = -1;
   for (int k = 0; k < this->numberOfParameterSweeps(); k++) {
      if (parameterSweepSize < 0) {
         parameterSweepSize = this->paramSweeps[k]->getNumValues();
      }
      else {
         if (parameterSweepSize != this->paramSweeps[k]->getNumValues()) {
            ErrorLog().printf(
                  "PVParams::setParameterSweepSize: all ParameterSweeps in the "
                  "parameters file must have the same number of entries.\n");
            abort();
         }
      }
   }
   if (parameterSweepSize < 0)
      parameterSweepSize = 0;
   return parameterSweepSize;
}

int PVParams::setParameterSweepValues(int n) {
   int status = PV_SUCCESS;
   // Set parameter sweeps
   if (n < 0 || n >= parameterSweepSize) {
      status = PV_FAILURE;
      return status;
   }
   for (int k = 0; k < this->numberOfParameterSweeps(); k++) {
      ParameterSweep *paramSweep = paramSweeps[k];
      SweepType type             = paramSweep->getType();
      const char *group_name     = paramSweep->getGroupName();
      const char *param_name     = paramSweep->getParamName();
      ParameterGroup *gp         = group(group_name);
      assert(gp != NULL);

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
   if (g == NULL) {
      if (worldRank == 0) {
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
   if (g == NULL) {
      if (worldRank == 0) {
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
      if (warnIfAbsent && worldRank == 0) {
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
   if (g == NULL) {
      if (worldRank == 0) {
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
   if (g == NULL) {
      if (worldRank == 0) {
         ErrorLog().printf("PVParams::value: couldn't find a group for %s\n", groupName);
      }
      return NULL;
   }
   const float *retval = g->arrayValues(paramName, size);
   if (retval == NULL) {
      assert(*size == 0);
      if (worldRank == 0) {
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
   if (g == NULL) {
      if (worldRank == 0) {
         ErrorLog().printf("PVParams::value: couldn't find a group for %s\n", groupName);
      }
      return NULL;
   }
   const double *retval = g->arrayValuesDbl(paramName, size);
   if (retval == NULL) {
      assert(*size == 0);
      if (worldRank == 0) {
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
            if (worldRank == 0 and warnIfAbsent == true) {
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
                  worldRank,
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
                  worldRank,
                  paramName,
                  groupName,
                  strerror(errno));
         }
         else if (!stringPresent(groupName, paramName)) {
            // Setting the param to NULL explicitly is allowed;
            // if the string parameter is not present at all, error out.
            if (worldRank == 0) {
               ErrorLog().printf(
                     "%s \"%s\": string parameter \"%s\" is required.\n",
                     groupKeywordFromName(groupName),
                     groupName,
                     paramName);
            }
            MPI_Barrier(icComm->globalCommunicator());
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
   if (g == NULL) {
      if (worldRank == 0) {
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
      if (warnIfAbsent && worldRank == 0) {
         WarnLog().printf(
               "No parameter string named \"%s\" in group \"%s\"\n", paramStringName, groupName);
      }
      return NULL;
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
   for (int i = 0; i < numGroups; i++) {
      if (strcmp(groupName, groups[i]->name()) == 0) {
         return groups[i];
      }
   }
   return NULL;
}

const char *PVParams::groupNameFromIndex(int index) {
   bool inbounds = index >= 0 && index < numGroups;
   return inbounds ? groups[index]->name() : NULL;
}

const char *PVParams::groupKeywordFromIndex(int index) {
   bool inbounds = index >= 0 && index < numGroups;
   return inbounds ? groups[index]->getGroupKeyword() : NULL;
}

const char *PVParams::groupKeywordFromName(const char *name) {
   const char *kw    = NULL;
   ParameterGroup *g = group(name);
   if (g != NULL) {
      kw = g->getGroupKeyword();
   }
   return kw;
}

/**
 * @keyword
 * @name
 */
void PVParams::addGroup(char *keyword, char *name) {
   assert((size_t)numGroups <= groupArraySize);

   // Verify that the new group's name is not an existing group's name
   for (int k = 0; k < numGroups; k++) {
      if (!strcmp(name, groups[k]->name())) {
         Fatal().printf("Rank %d process: group name \"%s\" duplicated\n", worldRank, name);
      }
   }

   if ((size_t)numGroups == groupArraySize) {
      groupArraySize += RESIZE_ARRAY_INCR;
      ParameterGroup **newGroups =
            (ParameterGroup **)malloc(groupArraySize * sizeof(ParameterGroup *));
      assert(newGroups);
      for (int k = 0; k < numGroups; k++) {
         newGroups[k] = groups[k];
      }
      free(groups);
      groups = newGroups;
   }

   groups[numGroups] = new ParameterGroup(name, stack, arrayStack, stringStack, worldRank);
   groups[numGroups]->setGroupKeyword(keyword);

   // the parameter group takes over control of the PVParams's stack and stringStack; make new ones.
   stack       = new ParameterStack(MAX_PARAMS);
   arrayStack  = new ParameterArrayStack(PARAMETERARRAYSTACK_INITIALCOUNT);
   stringStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);

   numGroups++;
}

void PVParams::addActiveParamSweep(const char *group_name, const char *param_name) {
   // Search for group_name and param_name in both ParameterSweep and BatchSweep list of objects
   for (int p = 0; p < numParamSweeps; p++) {
      if (strcmp(paramSweeps[p]->getGroupName(), group_name) == 0
          && strcmp(paramSweeps[p]->getParamName(), param_name) == 0) {
         Fatal().printf(
               "PVParams::addActiveParamSweep: Parameter sweep %s, %s already exists\n",
               group_name,
               param_name);
      }
   }

   activeParamSweep->setGroupAndParameter(group_name, param_name);
   ParameterSweep **newParamSweeps =
         (ParameterSweep **)calloc(numParamSweeps + 1, sizeof(ParameterSweep *));
   if (newParamSweeps == NULL) {
      Fatal().printf(
            "PVParams::action_parameter_sweep: unable to allocate memory for larger paramSweeps\n");
   }
   for (int k = 0; k < numParamSweeps; k++) {
      newParamSweeps[k] = paramSweeps[k];
   }
   free(paramSweeps);
   paramSweeps                 = newParamSweeps;
   paramSweeps[numParamSweeps] = activeParamSweep;
   numParamSweeps++;
   newActiveParamSweep();
}

int PVParams::warnUnread() {
   int status = PV_SUCCESS;
   for (int i = 0; i < numberOfGroups(); i++) {
      if (groups[i]->warnUnread() != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

bool PVParams::hasBeenRead(const char *group_name, const char *param_name) {
   ParameterGroup *g = group(group_name);
   if (g == NULL) {
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
   for (int i = 0; i < numberOfGroups(); i++) {
      if (groups[i]->clearHasBeenReadFlags() != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

void PVParams::handleUnnecessaryParameter(const char *group_name, const char *param_name) {
   if (present(group_name, param_name)) {
      if (worldRank == 0) {
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
      if (worldRank == 0) {
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
      if (worldRank == 0) {
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
            if (worldRank == 0) {
               ErrorLog().printf(
                     "%s \"%s\": Rank %d process unable to copy correct string value: %s.\n",
                     class_name,
                     group_name,
                     worldRank,
                     strerror(errno));
            }
         }
         if (params_value_i == nullptr) {
            status = PV_FAILURE;
            if (worldRank == 0) {
               ErrorLog().printf(
                     "%s \"%s\": Rank %d process unable to copy parameter string value: %s.\n",
                     class_name,
                     group_name,
                     worldRank,
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
         if (strcmp(params_value_i, correct_value_i) != 0) {
            status = PV_FAILURE;
            if (worldRank == 0) {
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
         if (worldRank == 0) {
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
         if (worldRank == 0) {
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
      MPI_Barrier(icComm->globalCommunicator());
      exit(EXIT_FAILURE);
   }
}

/**
 * @id
 * @val
 */
void PVParams::action_pvparams_directive(char *id, double val) {
   if (!strcmp(id, "debugParsing")) {
      debugParsing = (val != 0);
      if (worldRank == 0) {
         InfoLog(directiveMessage);
         directiveMessage.printf("debugParsing turned ");
         if (debugParsing) {
            directiveMessage.printf("on.\n");
         }
         else {
            directiveMessage.printf("off.\n");
         }
      }
   }
   else if (!strcmp(id, "disable")) {
      disable = (val != 0);
      if (worldRank == 0) {
         InfoLog(directiveMessage);
         directiveMessage.printf("Parsing params file ");
         if (disable) {
            directiveMessage.printf("disabled.\n");
         }
         else {
            directiveMessage.printf("enabled.\n");
         }
      }
   }
   else {
      if (worldRank == 0) {
         WarnLog().printf("Unrecognized directive %s = %f, skipping.\n", id, val);
      }
   }
}

/**
 * @keyword
 * @name
 */
void PVParams::action_parameter_group() {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().printf(
            "action_parameter_group: %s \"%s\" parsed successfully.\n",
            currGroupKeyword,
            currGroupName);
      InfoLog().flush();
   }
   // build a parameter group
   addGroup(currGroupKeyword, currGroupName);
}
void PVParams::action_parameter_group_name(char *keyword, char *name) {
   if (disable)
      return;
   // remove surrounding quotes
   int len       = strlen(++name);
   name[len - 1] = '\0';

   if (debugParsing && worldRank == 0) {
      InfoLog().printf(
            "action_parameter_group_name: %s \"%s\" parsed successfully.\n", keyword, name);
      InfoLog().flush();
   }
   currGroupKeyword = keyword;
   currGroupName    = name;
}

/**
 * @id
 * @val
 */
void PVParams::action_parameter_def(char *id, double val) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_def: %s = %lf\n", id, val);
      InfoLog().flush();
   }
   checkDuplicates(id);
   Parameter *p = new Parameter(id, val);
   stack->push(p);
}

void PVParams::action_parameter_def_overwrite(char *id, double val) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_def_overwrite: %s = %lf\n", id, val);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   char *param_name     = stripOverwriteTag(id);
   Parameter *currParam = NULL;
   for (int i = 0; i < stack->size(); i++) {
      Parameter *param = stack->peek(i);
      if (strcmp(param->name(), param_name) == 0) {
         currParam = param;
      }
   }
   if (!currParam) {
      for (int i = 0; i < arrayStack->size(); i++) {
         ParameterArray *arrayParam = arrayStack->peek(i);
         if (strcmp(arrayParam->name(), param_name) == 0) {
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
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_array: %s\n", id);
      InfoLog().flush();
   }
   currentParamArray->setName(id);
   assert(!strcmp(currentParamArray->name(), id));
   checkDuplicates(id);
   arrayStack->push(currentParamArray);
   currentParamArray = new ParameterArray(PARAMETERARRAYSTACK_INITIALCOUNT);
}

void PVParams::action_parameter_array_overwrite(char *id) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_array_overwrite: %s\n", id);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   char *param_name          = stripOverwriteTag(id);
   ParameterArray *origArray = NULL;
   for (int i = 0; i < arrayStack->size(); i++) {
      ParameterArray *arrayParam = arrayStack->peek(i);
      if (strcmp(arrayParam->name(), param_name) == 0) {
         origArray = arrayParam;
      }
   }
   if (!origArray) {
      for (int i = 0; i < stack->size(); i++) {
         Parameter *param = stack->peek(i);
         if (strcmp(param->name(), param_name) == 0) {
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
   for (int i = 0; i < currentParamArray->getArraySize(); i++) {
      origArray->pushValue(currentParamArray->peek(i));
   }
   assert(origArray->getArraySize() == currentParamArray->getArraySize());
   delete currentParamArray;
   currentParamArray = new ParameterArray(PARAMETERARRAYSTACK_INITIALCOUNT);
}

void PVParams::action_parameter_array_value(double val) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_array_value %lf\n", val);
   }
#ifdef NDEBUG
   currentParamArray->pushValue(val);
#else
   int sz            = currentParamArray->getArraySize();
   int newsize       = currentParamArray->pushValue(val);
   assert(newsize == sz + 1);
#endif // NDEBUG
}

void PVParams::action_parameter_string_def(const char *id, const char *stringval) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_string_def: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   checkDuplicates(id);
   char *param_value = stripQuotationMarks(stringval);
   assert(!stringval || param_value); // stringval can be null, but if stringval is not null,
   // param_value should also be non-null
   ParameterString *pstr = new ParameterString(id, param_value);
   stringStack->push(pstr);
   free(param_value);
}

void PVParams::action_parameter_string_def_overwrite(const char *id, const char *stringval) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_string_def_overwrite: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   char *param_name           = stripOverwriteTag(id);
   ParameterString *currParam = NULL;
   for (int i = 0; i < stringStack->size(); i++) {
      ParameterString *param = stringStack->peek(i);
      assert(param);
      if (strcmp(param->getName(), param_name) == 0) {
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
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_filename_def: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   checkDuplicates(id);
   char *param_value = stripQuotationMarks(stringval);
   assert(param_value);
   ParameterString *pstr = new ParameterString(id, param_value);
   free(param_value);
   stringStack->push(pstr);
}

void PVParams::action_parameter_filename_def_overwrite(const char *id, const char *stringval) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_filename_def_overwrite: %s = %s\n", id, stringval);
      InfoLog().flush();
   }
   // Search through current parameters for the id
   char *param_name           = stripOverwriteTag(id);
   ParameterString *currParam = NULL;
   for (int i = 0; i < stringStack->size(); i++) {
      ParameterString *param = stringStack->peek(i);
      assert(param);
      if (strcmp(param->getName(), param_name) == 0) {
         currParam = param;
      }
   }
   free(param_name);
   param_name = NULL;
   if (!currParam) {
      ErrorLog().printf("Overwrite: %s is not an existing parameter to overwrite.\n", id);
   }
   char *param_value = stripQuotationMarks(stringval);
   assert(param_value);
   currParam->setValue(param_value);
   free(param_value);
}

void PVParams::action_include_directive(const char *stringval) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_include_directive: including %s\n", stringval);
      InfoLog().flush();
   }
   // The include directive must be the first parameter in the group if defined
   if (stack->size() != 0 || arrayStack->size() != 0 || stringStack->size() != 0) {
      ErrorLog().printf(
            "Import of %s must be the first parameter specified in the group.\n", stringval);
      InfoLog().flush();
   }
   // Grab the parameter value
   char *param_value = stripQuotationMarks(stringval);
   // Grab the included group's ParameterGroup object
   ParameterGroup *includeGroup = NULL;
   for (int groupidx = 0; groupidx < numGroups; groupidx++) {
      // If strings are matching
      if (strcmp(groups[groupidx]->name(), param_value) == 0) {
         includeGroup = groups[groupidx];
      }
   }
   // If group not found
   if (!includeGroup) {
      ErrorLog().printf("Include: include group %s is not defined.\n", param_value);
   }
   // Check keyword of group
   if (strcmp(includeGroup->getGroupKeyword(), currGroupKeyword) != 0) {
      ErrorLog().printf(
            "Include: Cannot include group %s, which is a %s, into a %s. Group types must be the "
            "same.\n",
            param_value,
            includeGroup->getGroupKeyword(),
            currGroupKeyword);
   }
   free(param_value);
   // Load all stack values into current parameter group

   assert(stack->size() == 0);
   delete stack;
   stack = includeGroup->copyStack();

   assert(arrayStack->size() == 0);
   delete arrayStack;
   arrayStack = includeGroup->copyArrayStack();

   assert(stringStack->size() == 0);
   delete stringStack;
   stringStack = includeGroup->copyStringStack();
}

void PVParams::action_parameter_sweep_open(const char *groupname, const char *paramname) {
   if (disable)
      return;
   // strip quotation marks from groupname
   currSweepGroupName = stripQuotationMarks(groupname);
   assert(currSweepGroupName);
   currSweepParamName = strdup(paramname);
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf(
            "action_parameter_sweep_open: Sweep for group %s, parameter \"%s\" starting\n",
            groupname,
            paramname);
      InfoLog().flush();
   }
}

void PVParams::action_parameter_sweep_close() {
   if (disable)
      return;
   addActiveParamSweep(currSweepGroupName, currSweepParamName);
   if (debugParsing && worldRank == 0) {
      InfoLog().printf(
            "action_parameter_group: ParameterSweep for %s \"%s\" parsed successfully.\n",
            currSweepGroupName,
            currSweepParamName);
      InfoLog().flush();
   }
   // build a parameter group
   free(currSweepGroupName);
   free(currSweepParamName);
}

void PVParams::action_parameter_sweep_values_number(double val) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_parameter_sweep_values_number: %f\n", val);
      InfoLog().flush();
   }
   activeParamSweep->pushNumericValue(val);
}

void PVParams::action_parameter_sweep_values_string(const char *stringval) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_sweep_values_string: %s\n", stringval);
      InfoLog().flush();
   }
   char *string = stripQuotationMarks(stringval);
   assert(!stringval || string); // stringval can be null, but if stringval is not null, string
   // should also be non-null
   activeParamSweep->pushStringValue(string);
   free(string);
}

void PVParams::action_parameter_sweep_values_filename(const char *stringval) {
   if (disable)
      return;
   if (debugParsing && worldRank == 0) {
      InfoLog().flush();
      InfoLog().printf("action_sweep_values_filename: %s\n", stringval);
      InfoLog().flush();
   }
   char *filename = stripQuotationMarks(stringval);
   assert(filename);
   activeParamSweep->pushStringValue(filename);
   free(filename);
}

void PVParams::checkDuplicates(const char *paramName) {
   bool hasDuplicate = false;
   for (int k = 0; k < stack->size(); k++) {
      Parameter *parm = stack->peek(k);
      if (!strcmp(paramName, parm->name())) {
         ErrorLog().printf(
               "Rank %d process: The params group for %s \"%s\" duplicates "
               "parameter \"%s\".\n",
               worldRank,
               currGroupKeyword,
               currGroupName,
               paramName);
         hasDuplicate = true;
      }
   }
   for (int k = 0; k < arrayStack->size(); k++) {
      if (!strcmp(paramName, arrayStack->peek(k)->name())) {
         ErrorLog().printf(
               "Rank %d process: The params group for %s \"%s\" duplicates "
               "array parameter \"%s\".\n",
               worldRank,
               currGroupKeyword,
               currGroupName,
               paramName);
         hasDuplicate = true;
      }
   }
   for (int k = 0; k < stringStack->size(); k++) {
      if (!strcmp(paramName, stringStack->peek(k)->getName())) {
         ErrorLog().printf(
               "Rank %d process: The params group for %s \"%s\" duplicates "
               "string parameter \"%s\".\n",
               worldRank,
               currGroupKeyword,
               currGroupName,
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
   // beginning and end, return NULL.
   // It is the responsibility of the routine that calls stripQuotationMarks
   // to free the returned string to avoid a memory leak.
   if (s == NULL) {
      return NULL;
   }
   char *noquotes = NULL;
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
   char *notag = NULL;
   if (len >= 1 && s[0] == '@') {
      notag = (char *)calloc(len, sizeof(char));
      memcpy(notag, s + 1, len - 1);
      notag[len - 1] = '\0';
   }
   return notag;
}

} // close namespace PV block
