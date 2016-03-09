/*
 * PVParams.cpp
 *
 *  Created on: Nov 27, 2008
 *      Author: rasmussn
 */

#include "PVParams.hpp"
#include "../include/pv_common.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cmath> // nearbyint()
#include <climits> // INT_MIN

#define PARAMETERARRAY_INITIALSIZE 8
#define PARAMETERARRAYSTACK_INITIALCOUNT 5
#define PARAMETERSTRINGSTACK_INITIALCOUNT 5
#define PARAMETERSWEEP_INCREMENTCOUNT 10

// define for debug output
#define DEBUG_PARSING

#ifdef HAS_MAIN
extern FILE* yyin;
#endif // HAS_MAIN

/**
 * @yyin
 * @action_handler
 * @paramBuffer
 * @len
 */
int pv_parseParameters(PV::PVParams * action_handler, const char * paramBuffer, size_t len);

#ifdef HAS_MAIN
#define INITIALNUMGROUPS 20   // maximum number of groups
int main()
{
   PV_Stream pvstream = PV_fopen("parser/params.txt", "r", false);
   yyin = pvstream->fp;
   PV::PVParams * handler = new PV::PVParams(INITIAL_NUM_GROUPS);

   pv_parseParameters(handler);

   PV_fclose(pvstream);
   yyin = NULL;
   delete handler;

   return 0;
}
#endif // HAS_MAIN

namespace PV {

/**
 * @name
 * @value
 */
Parameter::Parameter(const char * name, double value)
{
   paramName  = strdup(name);
   paramValue = (float) value;
   paramDblValue = value;
   hasBeenReadFlag = false;
}

Parameter::~Parameter()
{
   free(paramName);
}

int Parameter::outputParam(FILE * fp, int indentation) {
   int status = PV_SUCCESS;
   for( int i=indentation; i>0; i-- ) fputc(' ', fp);
   fprintf(fp, "%s : %.17e", paramName, paramDblValue);
   if( paramDblValue == 1.0f ) fprintf(fp, " (true)");
   else if( paramDblValue == 1.0f ) fprintf(fp, " (false)");
   else if( paramDblValue == FLT_MAX ) fprintf(fp, " (infinity)");
   else if( paramDblValue == -FLT_MAX ) fprintf(fp, " (-infinity)");
   fprintf(fp, "\n");
   return status;
}

ParameterArray::ParameterArray(int initialSize) {
   hasBeenReadFlag = false;
   paramNameSet = false;
   paramName = strdup("Unnamed Parameter array");
   bufferSize = initialSize;
   arraySize = 0;
   values = NULL;
   if (bufferSize>0) {
      values = (float *) calloc(bufferSize,sizeof(float));
      valuesDbl = (double *) calloc(bufferSize, sizeof(double));
      if (values == NULL || valuesDbl == NULL) {
         fprintf(stderr, "ParameterArray error allocating memory for \"%s\"\n", name());
         abort();
      }
   }
}

ParameterArray::~ParameterArray() {
   free(paramName); paramName = NULL;
   free(values); values = NULL;
   free(valuesDbl); valuesDbl = NULL;
}

int ParameterArray::setName(const char * name) {
   int status = PV_SUCCESS;
   if (paramNameSet == false) {
      free(paramName); paramName = strdup(name);
      paramNameSet = true;
   }
   else {
      fprintf(stderr, "ParameterArray::setName called with \"%s\" but name is already set to \"%s\"\n", name, paramName);
      status = PV_FAILURE;
   }
   return status;
}

int ParameterArray::pushValue(double value) {
   assert(bufferSize>=arraySize);
   if (bufferSize==arraySize) {
      bufferSize += PARAMETERARRAY_INITIALSIZE;
      float * new_values = (float *) calloc(bufferSize,sizeof(float));
      if (new_values == NULL) {
         fprintf(stderr, "ParameterArray::pushValue error increasing array \"%s\" to %d values\n", name(), arraySize+1);
         abort();
      }
      memcpy(new_values, values, sizeof(float)*arraySize);
      free(values); values = new_values;
      double * new_values_dbl = (double *) calloc(bufferSize,sizeof(double));
      if (new_values == NULL) {
         fprintf(stderr, "ParameterArray::pushValue error increasing array \"%s\" to %d values\n", name(), arraySize+1);
         abort();
      }
      memcpy(new_values_dbl, valuesDbl, sizeof(double)*arraySize);
      free(valuesDbl); valuesDbl = new_values_dbl;
   }
   assert(arraySize<bufferSize);
   valuesDbl[arraySize] = value;
   values[arraySize] = (float) value;
   arraySize++;
   return arraySize;
}

int ParameterArray::outputString(FILE * fp, int indentation) {
   int status = PV_SUCCESS;
   for( int i=indentation; i>0; i--) fputc(' ', fp);
   fprintf(fp, "%s : Values:\n", paramName);
   int sz = 0;
   const double * vals = getValuesDbl(&sz);
   for (int j=0; j<sz; j++) {
      for (int i=indentation; i>0; i--) fputc(' ', fp);
      fprintf(fp, "    value %d = %f\n", j, vals[j]);
   }
   return status;
}

ParameterArray* ParameterArray::copyParameterArray(){
   ParameterArray* returnPA = new ParameterArray(bufferSize);
   int status = returnPA->setName(paramName);
   assert(status == PV_SUCCESS);
   for(int i = 0; i < arraySize; i++){
      returnPA->pushValue(valuesDbl[i]);
   }
   return returnPA;
}

/**
 * @name
 * @value
 */
ParameterString::ParameterString(const char * name, const char * value)
{
   paramName = name ? strdup(name) : NULL;
   paramValue = value ? strdup(value) : NULL;
   hasBeenReadFlag = false;
}

ParameterString::~ParameterString()
{
   free(paramName);
   free(paramValue);
}

int ParameterString::outputString(FILE * fp, int indentation) {
   int status = PV_SUCCESS;
   for( int i=indentation; i>0; i--) fputc(' ', fp);
   fprintf(fp, "%s : \"%s\"\n", paramName, paramValue);
   return status;
}

/**
 * @maxCount
 */
ParameterStack::ParameterStack(int maxCount)
{
   this->maxCount = maxCount;
   count = 0;
   parameters = (Parameter **) malloc(maxCount * sizeof(Parameter *));
}

ParameterStack::~ParameterStack()
{
   for (int i = 0; i < count; i++) {
      delete parameters[i];
   }
   free(parameters);
}

/**
 * @param
 */
int ParameterStack::push(Parameter* param)
{
   assert(count < maxCount);
   parameters[count++] = param;
   return 0;
}

Parameter * ParameterStack::pop()
{
   assert(count > 0);
   return parameters[count--];
}

int ParameterStack::outputStack(FILE * fp, int indentation) {
   int status = PV_SUCCESS;
   for( int i=indentation; i>0; i-- ) {
      fputc(' ', fp);
   }
   fprintf(fp, "// numerical parameters\n");
   for( int s=0; s<count; s++ ) {
      if( parameters[s]->outputParam(fp, indentation) != PV_SUCCESS ) status = PV_FAILURE;
   }
   return status;
}

ParameterArrayStack::ParameterArrayStack(int initialCount)
{
   allocation = initialCount;
   count = 0;
   parameterArrays = NULL;
   if (initialCount > 0) {
      parameterArrays = (ParameterArray **) calloc(allocation, sizeof(ParameterArray *));
      if (parameterArrays == NULL) {
         fprintf(stderr, "ParameterArrayStack unable to allocate %d parameter arrays\n", initialCount);
         abort();
      }
   }
}

ParameterArrayStack::~ParameterArrayStack() {
   for (int k=0; k<count; k++) {
      delete parameterArrays[k]; parameterArrays[k] = NULL;
   }
   free(parameterArrays); parameterArrays = NULL;
}

int ParameterArrayStack::push(ParameterArray * array) {
   assert(count<=allocation);
   if (count == allocation) {
      int newallocation = allocation + RESIZE_ARRAY_INCR;
      ParameterArray ** newParameterArrays = (ParameterArray **) malloc( newallocation*sizeof(ParameterArray *) );
      if( !newParameterArrays ) return PV_FAILURE;
      for( int i=0; i<count; i++ ) {
         newParameterArrays[i] = parameterArrays[i];
      }
      allocation = newallocation;
      free(parameterArrays); parameterArrays = newParameterArrays;
   }
   assert( count < allocation );
   parameterArrays[count] = array;
   count++;
   return PV_SUCCESS;
}

int ParameterArrayStack::outputStack(FILE * fp, int indentation) {
   int status = PV_SUCCESS;
   for( int i=indentation; i>0; i-- ) {
      fputc(' ', fp);
   }
   fprintf(fp, "// array parameters\n");
   for( int s=0; s<count; s++ ) {
      if( parameterArrays[s]->outputString(fp, indentation) != PV_SUCCESS ) status = PV_FAILURE;
   }
   return status;
}


/*
 * initialCount
 */
ParameterStringStack::ParameterStringStack(int initialCount)
{
   allocation = initialCount;
   count = 0;
   parameterStrings = (ParameterString **) calloc( allocation, sizeof(ParameterString *));
}

ParameterStringStack::~ParameterStringStack()
{
   for( int i=0; i<count; i++ ) {
      delete parameterStrings[i];
   }
   free(parameterStrings);
}

/*
 * @param
 */
int ParameterStringStack::push(ParameterString * param)
{
   assert( count <= allocation );
   if( count == allocation ) {
      int newallocation = allocation + RESIZE_ARRAY_INCR;
      ParameterString ** newparameterStrings = (ParameterString **) malloc( newallocation*sizeof(ParameterString *) );
      if( !newparameterStrings ) return PV_FAILURE;
      for( int i=0; i<count; i++ ) {
         newparameterStrings[i] = parameterStrings[i];
      }
      allocation = newallocation;
      free(parameterStrings);
      parameterStrings = newparameterStrings;
   }
   assert( count < allocation );
   parameterStrings[count++] = param;
   return PV_SUCCESS;
}

ParameterString * ParameterStringStack::pop()
{
   if(count > 0) {
      return parameterStrings[count--];
   }
   else return NULL;
}

const char * ParameterStringStack::lookup(const char * targetname)
{
   const char * result = NULL;
   for( int i=0; i<count; i++ ) {
      if( !strcmp(parameterStrings[i]->getName(),targetname) ) {
         result = parameterStrings[i]->getValue();
      }
   }
   return result;
}

int ParameterStringStack::outputStack(FILE * fp, int indentation) {
   int status = PV_SUCCESS;
   for( int i=indentation; i>0; i-- ) {
      fputc(' ', fp);
   }
   fprintf(fp, "// string parameters\n");
   for( int s=0; s<count; s++ ) {
      if( parameterStrings[s]->outputString(fp, indentation) != PV_SUCCESS ) status = PV_FAILURE;
   }
   return status;
}


/**
 * @name
 * @stack
 * @string_stack
 * @rank
 */
ParameterGroup::ParameterGroup(char * name, ParameterStack * stack, ParameterArrayStack * array_stack, ParameterStringStack * string_stack, int rank)
{
   this->groupName = strdup(name);
   this->groupKeyword = NULL;
   this->stack     = stack;
   this->arrayStack = array_stack;
   this->stringStack = string_stack;
   this->processRank = rank;
}

ParameterGroup::~ParameterGroup()
{
   free(groupName); groupName = NULL;
   free(groupKeyword); groupKeyword = NULL;
   delete stack; stack = NULL;
   delete arrayStack; arrayStack = NULL;
   delete stringStack; stringStack = NULL;
}

int ParameterGroup::setGroupKeyword(const char * keyword) {
   if( groupKeyword == NULL ) {
      size_t keywordlen = strlen(keyword);
      groupKeyword = (char *) malloc(keywordlen+1);
      if( groupKeyword ) {
          strcpy(groupKeyword, keyword);
      }
   }
   return groupKeyword == NULL ? PV_FAILURE : PV_SUCCESS;
}

int ParameterGroup::setStringStack(ParameterStringStack * stringStack) {
   this->stringStack = stringStack;
   // ParameterGroup::setStringStack takes ownership of the stringStack;
   // i.e. it will delete it when the ParameterGroup is deleted.
   // You shouldn't use a stringStack after calling this routine with it.
   // Instead, query it with ParameterGroup::stringPresent and
   // ParameterGroup::stringValue methods.
   return stringStack==NULL ? PV_FAILURE : PV_SUCCESS;
}

/**
 * @name
 */
int ParameterGroup::present(const char * name)
{
   int count = stack->size();
   for (int i = 0; i < count; i++) {
      Parameter * p = stack->peek(i);
      if (strcmp(name, p->name()) == 0) {
         return 1;  // string is present
      }
   }
   return 0;  // string not present
}

/**
 * @name
 */
double ParameterGroup::value(const char * name)
{
   int count = stack->size();
   for (int i = 0; i < count; i++) {
      Parameter * p = stack->peek(i);
      if (strcmp(name, p->name()) == 0) {
         return p->value();
      }
   }
   fprintf(stderr, "PVParams::ParameterGroup::value: ERROR, couldn't find a value for %s"
                   " in group %s\n", name, groupName);
   exit(1);
}

bool ParameterGroup::arrayPresent(const char * name) {
   int array_found = 0;
   int count = arrayStack->size();
   for (int i=0; i<count; i++) {
      ParameterArray * p = arrayStack->peek(i);
      if (strcmp(name, p->name())==0) {
         array_found = 1; // string is present
         break;
      }
   }
   return array_found;
}

const float * ParameterGroup::arrayValues(const char * name, int * size) {
   int count = arrayStack->size();
   *size = 0;
   const float * v = NULL;
   ParameterArray * p = NULL;
   for (int i=0; i<count; i++) {
      p = arrayStack->peek(i);
      if (strcmp(name, p->name())==0) {
         v = p->getValues(size);
         break;
      }
   }
   if (!v) {
      Parameter * q = NULL;
      for (int i=0; i<stack->size(); i++) {
         Parameter * q1 = stack->peek(i);
         assert(q1);
         if (strcmp(name, q1->name())==0) {
            q = q1;
            break;
         }
      }
      if (q) {
         v = q->valuePtr();
         *size = 1;
      }
   }
   return v;
}

const double * ParameterGroup::arrayValuesDbl(const char * name, int * size) {
   int count = arrayStack->size();
   *size = 0;
   const double * v = NULL;
   ParameterArray * p = NULL;
   for (int i=0; i<count; i++) {
      p = arrayStack->peek(i);
      if (strcmp(name, p->name())==0) {
         v = p->getValuesDbl(size);
         break;
      }
   }
   if (!v) {
      Parameter * q = NULL;
      for (int i=0; i<stack->size(); i++) {
         Parameter * q1 = stack->peek(i);
         assert(q1);
         if (strcmp(name, q1->name())==0) {
            q = q1;
            break;
         }
      }
      if (q) {
         v = q->valueDblPtr();
         *size = 1;
      }
   }
   return v;
}

int ParameterGroup::stringPresent(const char * stringName) {
   // not really necessary, as stringValue returns NULL if the
   // string is not found, but included on the analogy with
   // value and present methods for floating-point parameters
   if( !stringName ) return 0;
   int count = stringStack->size();
   for( int i=0; i<count; i++) {
      ParameterString * pstr = stringStack->peek(i);
      assert(pstr);
      if( !strcmp( stringName, pstr->getName() ) ) {
         return 1;  // string is present
      }
   }
   return 0;  // string not present
}

const char * ParameterGroup::stringValue(const char * stringName ) {
   if( !stringName ) return NULL;
   int count = stringStack->size();
   for( int i=0; i<count; i++ ) {
      ParameterString * pstr = stringStack->peek(i);
      assert(pstr);
      if( !strcmp( stringName, pstr->getName() ) ) {
         return pstr->getValue();
      }
   }
   return NULL;
}

int ParameterGroup::warnUnread() {
   int status = PV_SUCCESS;
   int count;
   count = stack->size();
   for( int i=0; i<count; i++ ) {
      Parameter * p = stack->peek(i);
      if( !p->hasBeenRead() ) {
         if( processRank==0 ) fprintf(stderr,"Parameter group \"%s\": parameter \"%s\" has not been read.\n", name(), p->name());
         status = PV_FAILURE;
      }
   }
   count = arrayStack->size();
   for (int i=0; i<count; i++) {
      ParameterArray * parr = arrayStack->peek(i);
      if (!parr->hasBeenRead()) {
         if (processRank==0) fprintf(stderr,"Parameter group \"%s\": array parameter \"%s\" has not been read.\n", name(), parr->name());
      }
   }
   count = stringStack->size();
   for( int i=0; i<count; i++ ) {
      ParameterString * pstr = stringStack->peek(i);
      if( !pstr->hasBeenRead() ) {
         if( processRank==0 ) fprintf(stderr,"Parameter group \"%s\": string parameter \"%s\" has not been read.\n", name(), pstr->getName());
         status = PV_FAILURE;
      }
   }
   return status;
}

bool ParameterGroup::hasBeenRead(const char * paramName) {
   int count;
   count = stack->size();
   for( int i=0; i<count; i++ ) {
      Parameter * p = stack->peek(i);
      if( !strcmp(p->name(),paramName) ) {
         return p->hasBeenRead();
      }
   }
   count = arrayStack->size();
   for (int i=0; i<count; i++) {
      ParameterArray * parr = arrayStack->peek(i);
      if (!strcmp(parr->name(), paramName)) {
         return parr->hasBeenRead();
      }
   }
   count = stringStack->size();
   for( int i=0; i<count; i++ ) {
      ParameterString * pstr = stringStack->peek(i);
      if( !strcmp(pstr->getName(), paramName) ) {
         return pstr->hasBeenRead();
      }
   }
   return false;
}

int ParameterGroup::clearHasBeenReadFlags() {
   int status = PV_SUCCESS;
   int count = stack->size();
   for( int i=0; i<count; i++ ) {
      Parameter * p = stack->peek(i);
      p->clearHasBeenRead();
   }
   count = arrayStack->size();
   for (int i=0; i<count; i++) {
      ParameterArray * parr = arrayStack->peek(i);
      parr->clearHasBeenRead();
   }
   count = stringStack->size();
   for( int i=0; i<count; i++ ) {
      ParameterString * pstr = stringStack->peek(i);
      pstr->clearHasBeenRead();
   }
   return status;
}

int ParameterGroup::outputGroup(FILE * fp) {
   int status = PV_SUCCESS;
   fprintf(fp, "%s \"%s\":\n", groupKeyword, groupName);
   int indentation = 4;
   if( stack->outputStack(fp, indentation) != PV_SUCCESS ) status = PV_FAILURE;
   if (arrayStack->outputStack(fp, indentation) != PV_SUCCESS) status = PV_FAILURE;
   if( stringStack->outputStack(fp, indentation) != PV_SUCCESS ) status = PV_FAILURE;
   return status;
}

int ParameterGroup::pushNumerical(Parameter * param) {
   return stack->push(param);
}

int ParameterGroup::pushString(ParameterString * param) {
   return stringStack->push(param);
}

int ParameterGroup::setValue(const char * param_name, double value) {
   int status = PV_SUCCESS;
   int count = stack->size();
   for (int i = 0; i < count; i++) {
      Parameter * p = stack->peek(i);
      if (strcmp(param_name, p->name()) == 0) {
         p->setValue(value);
         return PV_SUCCESS;
      }
   }
   fprintf(stderr, "PVParams::ParameterGroup::setValue: ERROR, couldn't find parameter %s"
         " in group \"%s\"\n", param_name, name());
   exit(PV_FAILURE);

   return status;
}

int ParameterGroup::setStringValue(const char * param_name, const char * svalue) {
   int status = PV_SUCCESS;
   int count = stringStack->size();
   for (int i = 0; i < count; i++) {
      ParameterString * p = stringStack->peek(i);
      if (strcmp(param_name, p->getName()) == 0) {
         p->setValue(svalue);
         return PV_SUCCESS;
      }
   }
   fprintf(stderr, "PVParams::ParameterGroup::setStringValue: ERROR, couldn't find a string value for %s"
         " in group \"%s\"\n", param_name, name());
   exit(PV_FAILURE);

   return status;
}

/**
 * A function to return a copy of the parameter group's stack.
 */
ParameterStack* ParameterGroup::copyStack(){
   ParameterStack* returnStack = new ParameterStack(MAX_PARAMS);
   for(int i = 0; i < stack->size(); i++){
      returnStack->push(stack->peek(i)->copyParameter());
   }
   return returnStack;
}

/**
 * A function to return a copy of the parameter group's stack.
 */
ParameterArrayStack* ParameterGroup::copyArrayStack(){
   ParameterArrayStack* returnStack = new ParameterArrayStack(PARAMETERARRAYSTACK_INITIALCOUNT);
   for(int i = 0; i < arrayStack->size(); i++){
      returnStack->push(arrayStack->peek(i)->copyParameterArray());
   }
   return returnStack;
}

/**
 * A function to return a copy of the parameter group's stack.
 */
ParameterStringStack* ParameterGroup::copyStringStack(){
   ParameterStringStack* returnStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);
   for(int i = 0; i < stringStack->size(); i++){
      returnStack->push(stringStack->peek(i)->copyParameterString());
   }
   return returnStack;
}


ParameterSweep::ParameterSweep() {
   groupName = NULL;
   paramName = NULL;
   numValues = 0;
   currentBufferSize = 0;
   type = SWEEP_UNDEF;
   valuesNumber = NULL;
   valuesString = NULL;
}

ParameterSweep::~ParameterSweep() {
   free(groupName); groupName = NULL;
   free(paramName); paramName = NULL;
   free(valuesNumber); valuesNumber = NULL;
   if (valuesString!=NULL) {
      for (int k=0; k<numValues; k++) {
         free(valuesString[k]);
      }
      free(valuesString); valuesString = NULL;
   }
}

int ParameterSweep::setGroupAndParameter(const char * groupname, const char * parametername) {
   int status = PV_SUCCESS;
   if (groupName != NULL || paramName != NULL) {
      fprintf(stderr,"ParameterSweep::setGroupParameter error:");
      if (groupName != NULL) {
         fprintf(stderr, " groupName has already been set to \"%s\".", groupName);
      }
      if (paramName != NULL) {
         fprintf(stderr, " paramName has already been set to \"%s\".", paramName);
      }
      fprintf(stderr, "\n");
      status = PV_FAILURE;
   }
   else {
      groupName = strdup(groupname);
      paramName = strdup(parametername);
      //Check for duplicates

   }
   return status;
}

int ParameterSweep::pushNumericValue(double val) {
   int status = PV_SUCCESS;
   if (numValues==0) {
      type = SWEEP_NUMBER;
   }
   assert(type==SWEEP_NUMBER);
   assert(valuesString == NULL);

   assert(numValues <= currentBufferSize);
   if (numValues == currentBufferSize) {
      currentBufferSize += PARAMETERSWEEP_INCREMENTCOUNT;
      double * newValuesNumber = (double *) calloc(currentBufferSize, sizeof(double));
      if (newValuesNumber == NULL) {
         fprintf(stderr, "ParameterSweep:pushNumericValue error: unable to allocate memory\n");
         status = PV_FAILURE;
         abort();
      }
      for (int k=0; k<numValues; k++) {
         newValuesNumber[k] = valuesNumber[k];
      }
      free(valuesNumber);
      valuesNumber = newValuesNumber;
   }
   valuesNumber[numValues] = val;
   numValues++;
   return status;
}

int ParameterSweep::pushStringValue(const char * sval) {
   int status = PV_SUCCESS;
   if (numValues==0) {
      type = SWEEP_STRING;
   }
   assert(type==SWEEP_STRING);
   assert(valuesNumber == NULL);

   assert(numValues <= currentBufferSize);
   if (numValues == currentBufferSize) {
      currentBufferSize += PARAMETERSWEEP_INCREMENTCOUNT;
      char ** newValuesString = (char **) calloc(currentBufferSize, sizeof(char *));
      if (newValuesString == NULL) {
         fprintf(stderr, "ParameterSweep:pushStringValue error: unable to allocate memory\n");
         status = PV_FAILURE;
         abort();
      }
      for (int k=0; k<numValues; k++) {
         newValuesString[k] = valuesString[k];
      }
      free(valuesString);
      valuesString = newValuesString;
   }
   valuesString[numValues] = strdup(sval);
   numValues++;
   return status;
}

int ParameterSweep::getNumericValue(int n, double * val) {
   int status = PV_SUCCESS;
   assert(valuesNumber != NULL);
   if ( type != SWEEP_NUMBER || n<0 || n >= numValues ) {
      status = PV_FAILURE;
   }
   else {
      *val = valuesNumber[n];
   }
   return status;
}

const char * ParameterSweep::getStringValue(int n) {
   char * str = NULL;
   assert(valuesString != NULL);
   if ( type == SWEEP_STRING && n>=0 && n < numValues ) {
      str = valuesString[n];
   }
   return str;
}

/**
 * @filename
 * @initialSize
 * @icComm
 */
PVParams::PVParams(const char * filename, size_t initialSize, InterColComm* inIcComm)
{
   this->icComm = inIcComm;
   initialize(initialSize);
   parseFile(filename);
}

/*
 * @initialSize
 * @icComm
 */
PVParams::PVParams(size_t initialSize, InterColComm* inIcComm)
{
   this->icComm = inIcComm;
   initialize(initialSize);
}

/*
 * @buffer
 * @bufferLength
 * @initialSize
 * @icComm
 */
PVParams::PVParams(const char * buffer, long int bufferLength, size_t initialSize, InterColComm* inIcComm)
{
   this->icComm = inIcComm;
   initialize(initialSize);
   parseBuffer(buffer, bufferLength);
}

PVParams::~PVParams()
{
   for( int i=0; i<numGroups; i++) {
      delete groups[i];
   }
   free(groups);
   delete currentParamArray; currentParamArray = NULL;
   delete stack;
   delete arrayStack;
   delete stringStack;
   delete this->activeParamSweep;
   delete this->activeBatchSweep;
   for (int i=0; i<numParamSweeps; i++) {
      delete paramSweeps[i];
   }
   for (int i=0; i<numBatchSweeps; i++) {
      delete batchSweeps[i];
   }
   free(paramSweeps); paramSweeps = NULL;
   free(batchSweeps); batchSweeps = NULL;
}

/*
 * @initialSize
 */
int PVParams::initialize(size_t initialSize) {
   this->numGroups = 0;
   groupArraySize = initialSize;
   //this->icComm = icComm;
   //Get world rank and size
   MPI_Comm_rank(icComm->globalCommunicator(), &worldRank);
   MPI_Comm_size(icComm->globalCommunicator(), &worldSize);
   
   groups = (ParameterGroup **) malloc(initialSize * sizeof(ParameterGroup *));
   stack = new ParameterStack(MAX_PARAMS);
   arrayStack = new ParameterArrayStack(PARAMETERARRAYSTACK_INITIALCOUNT);
   stringStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);

   currentParamArray = new ParameterArray(PARAMETERARRAYSTACK_INITIALCOUNT);

   numParamSweeps = 0;
   numBatchSweeps = 0;
   paramSweeps = NULL;
   batchSweeps = NULL;
   newActiveParamSweep();
   newActiveBatchSweep();
#ifdef DEBUG_PARSING
   debugParsing = true;
#else
   debugParsing = false;
#endif//DEBUG_PARSING
   disable = false;

   return ( groups && stack && stringStack && activeParamSweep && activeBatchSweep /* && fnstack */ ) ? PV_SUCCESS : PV_FAILURE;
}

int PVParams::newActiveParamSweep() {
   int status = PV_SUCCESS;
   activeParamSweep = new ParameterSweep();
   if (activeParamSweep == NULL) {
      fprintf(stderr, "PVParams::newActiveParamSweep: unable to create activeParamSweep");
      status = PV_FAILURE;
      abort();
   }
   return status;
}

int PVParams::newActiveBatchSweep() {
   int status = PV_SUCCESS;
   activeBatchSweep = new ParameterSweep();
   if (activeBatchSweep == NULL) {
      fprintf(stderr, "PVParams::newActiveBatchSweep: unable to create activeBatchSweep");
      status = PV_FAILURE;
      abort();
   }
   return status;
}

int PVParams::parseFile(const char * filename) {
   int rootproc = 0;
   char * paramBuffer = NULL;
   size_t bufferlen;
   if( worldRank == rootproc ) {
      if( filename == NULL ) {
         fprintf(stderr, "PVParams::parseFile: filename was null.\n");
         exit(ENOENT);
      }
      struct stat filestatus;
      if( stat(filename, &filestatus) ) {
         fprintf(stderr, "PVParams::parseFile ERROR getting status of file \"%s\": %s\n", filename, strerror(errno));
         exit(errno);
      }
      if( filestatus.st_mode & S_IFDIR ) {
         fprintf(stderr, "PVParams::parseFile ERROR: specified file \"%s\" is a directory.\n", filename);
         exit(EISDIR);
      }
      PV_Stream * paramstream = PV_fopen(filename, "r", false/*verifyWrites*/);
      if( paramstream == NULL ) {
         fprintf(stderr, "PVParams::parseFile ERROR opening file \"%s\": %s\n", filename, strerror(errno));
         exit(errno);
      }
      if( PV_fseek(paramstream, 0, SEEK_END) != 0 ) {
         fprintf(stderr, "PVParams::parseFile ERROR seeking end of file \"%s\": %s\n", filename, strerror(errno));
         exit(errno);
      }
      bufferlen = (size_t) getPV_StreamFilepos(paramstream);
      paramBuffer = (char *) malloc(bufferlen);
      if( paramBuffer == NULL ) {
         fprintf(stderr, "PVParams::parseFile: Rank %d process unable to allocate memory for params buffer\n", rootproc);
         exit(ENOMEM);
      }
      PV_fseek(paramstream, 0L, SEEK_SET);
      if( PV_fread(paramBuffer,1, (unsigned long int) bufferlen, paramstream) != bufferlen) {
         fprintf(stderr, "PVParams::parseFile: ERROR reading params file \"%s\"", filename);
         exit(EIO);
      }
      PV_fclose(paramstream);
#ifdef PV_USE_MPI
      int sz = worldSize;
      for( int i=0; i<sz; i++ ) {
         if( i==rootproc ) continue;
         MPI_Send(paramBuffer, (int) bufferlen, MPI_CHAR, i, 31, icComm->globalCommunicator());
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
      bufferlen = (size_t) count;
      paramBuffer = (char *) malloc(bufferlen);
      if( paramBuffer == NULL ) {
         fprintf(stderr, "PVParams::parseFile: Rank %d process unable to allocate memory for params buffer\n", worldRank);
         abort();
      }
      MPI_Recv(paramBuffer, (int) bufferlen, MPI_CHAR, rootproc, 31, icComm->globalCommunicator(), MPI_STATUS_IGNORE);
#endif // PV_USE_MPI
   }

   int status = parseBuffer(paramBuffer, bufferlen);
   free(paramBuffer);
   return status;
}

bool PVParams::hasSweepValue(const char* inParamName){
   bool out = false;
   const char * group_name;
   for (int k=0; k<numberOfParameterSweeps(); k++) {
      ParameterSweep * sweep = paramSweeps[k];
      group_name = sweep->getGroupName();
      const char * param_name = sweep->getParamName();
      ParameterGroup * gp = group(group_name);
      if (gp == NULL) {
         fprintf(stderr, "PVParams::parseBuffer error: ParameterSweep %d (zero-indexed) refers to non-existent group \"%s\"\n", k, group_name);
         exit(EXIT_FAILURE);
      }
      if ( !strcmp(gp->getGroupKeyword(),"HyPerCol") && !strcmp(param_name, inParamName)) {
         out = true;
         break;
      }
   }

   if(!out){
      for (int k=0; k<numberOfBatchSweeps(); k++) {
         ParameterSweep * sweep = batchSweeps[k];
         group_name = sweep->getGroupName();
         const char * param_name = sweep->getParamName();
         ParameterGroup * gp = group(group_name);
         if (gp == NULL) {
            fprintf(stderr, "PVParams::parseBuffer error: BatchSweep %d (zero-indexed) refers to non-existent group \"%s\"\n", k, group_name);
            exit(EXIT_FAILURE);
         }
         if ( !strcmp(gp->getGroupKeyword(),"HyPerCol") && !strcmp(param_name, inParamName) ) {
            out = true;
            break;
         }
      }
   }
   return out;
}

int PVParams::parseBuffer(char const * buffer, long int bufferLength) {
   // Assumes that each MPI process has the same contents in buffer.

   fflush(stdout);
   //This is where it calls the scanner and parser
   parseStatus = pv_parseParameters(this, buffer, bufferLength);
   if( parseStatus != 0 ) {
      fprintf(stderr, "Rank %d process: pv_parseParameters failed with return value %d\n", worldRank, parseStatus);
   }
   fflush(stdout);

   setParameterSweepSize(); // Need to set sweepSize here, because if the outputPath sweep needs to be created we need to know the size.
   setBatchSweepSize(); // Need to set sweepSize here, because if the outputPath sweep needs to be created we need to know the size.

   // If there is at least one ParameterSweep  and none of them set outputPath, create a parameterSweep that does set outputPath.


   //If both parameterSweep and batchSweep is set, must autoset output path, as there is no way to specify both paramSweep and batchSweep
   if(numberOfParameterSweeps() > 0 && numberOfBatchSweeps() > 0){
      fprintf(stderr, "PVParams::simultaneous batchSweep and parameterSweep not supported yet.\n");
      abort();
   }
   if (numberOfParameterSweeps() > 0) {
      if (!hasSweepValue("outputPath")) {
         const char * hypercolgroupname = NULL;
         const char * outputPathName = NULL;
         for (int g=0; g<numGroups; g++) {
            if (groups[g]->getGroupKeyword(),"HyPerCol") {
               hypercolgroupname = groups[g]->name();
               outputPathName = groups[g]->stringValue("outputPath");
               if(outputPathName == NULL){
                  fprintf(stderr, "PVParams::outputPath must be specified if parameterSweep does not sweep over outputPath\n");
                  abort();
               }
               break;
            }
         }
         if (hypercolgroupname == NULL) {
            fprintf(stderr, "PVParams::parseBuffer error: no HyPerCol group\n");
            abort();
         }

         char dummy;
         int lenserialno = snprintf(&dummy, 0, "%d", parameterSweepSize-1);
         int len = snprintf(&dummy, 0, "%s/paramsweep_%0*d/", outputPathName, lenserialno, parameterSweepSize-1)+1;
         char * outputPathStr = (char *) calloc(len, sizeof(char));
         if (outputPathStr == NULL) abort();
         for (int i=0; i<parameterSweepSize; i++) {
            int chars_needed = snprintf(outputPathStr, len, "%s/paramsweep_%0*d/", outputPathName, lenserialno, i);
            assert(chars_needed < len);
            activeParamSweep->pushStringValue(outputPathStr);
         }
         free(outputPathStr); outputPathStr = NULL;
         addActiveParamSweep(hypercolgroupname, "outputPath");
      }

      if(!hasSweepValue("checkpointWriteDir")){
         const char * hypercolgroupname = NULL;
         const char * checkpointWriteDir = NULL;
         for (int g=0; g<numGroups; g++) {
            if (groups[g]->getGroupKeyword(),"HyPerCol") {
               hypercolgroupname = groups[g]->name();
               checkpointWriteDir = groups[g]->stringValue("checkpointWriteDir");
               //checkpointWriteDir can be NULL if checkpointWrite is set to false
               break;
            }
         }
         if (hypercolgroupname == NULL) {
            fprintf(stderr, "PVParams::parseBuffer error: no HyPerCol group\n");
            abort();
         }
         if(checkpointWriteDir){
            char dummy;
            int lenserialno = snprintf(&dummy, 0, "%d", parameterSweepSize-1);
            int len = snprintf(&dummy, 0, "%s/paramsweep_%0*d/", checkpointWriteDir, lenserialno, parameterSweepSize-1)+1;
            char* checkpointPathStr = (char *) calloc(len, sizeof(char));
            if (checkpointPathStr == NULL) abort();
            for (int i=0; i<parameterSweepSize; i++) {
               int chars_needed = snprintf(checkpointPathStr, len, "%s/paramsweep_%0*d/", checkpointWriteDir, lenserialno, i);
               assert(chars_needed < len);
               activeParamSweep->pushStringValue(checkpointPathStr);
            }
            free(checkpointPathStr); checkpointPathStr = NULL;
            addActiveParamSweep(hypercolgroupname, "checkpointWriteDir");
         }
      }
   }

   if(icComm->numCommBatches() > 1){
      //This checks if there is a batch sweep of outputPath
      if (!hasSweepValue("outputPath")) {
         const char * hypercolgroupname = NULL;
         const char * outputPathName = NULL;
         for (int g=0; g<numGroups; g++) {
            if (groups[g]->getGroupKeyword(),"HyPerCol") {
               hypercolgroupname = groups[g]->name();
               outputPathName = groups[g]->stringValue("outputPath");
               if(outputPathName == NULL){
                  fprintf(stderr, "PVParams::outputPath must be specified if batchSweep does not sweep over outputPath\n");
                  abort();
               }
               break;
            }
         }
         if (hypercolgroupname == NULL) {
            fprintf(stderr, "PVParams::parseBuffer error: no HyPerCol group\n");
            abort();
         }
         char dummy;
         int lenserialno = snprintf(&dummy, 0, "%d", batchSweepSize-1);
         int len = snprintf(&dummy, 0, "%s/batchsweep_%0*d/", outputPathName, lenserialno, icComm->numCommBatches()-1)+1;
         char * outputPathStr = (char *) calloc(len, sizeof(char));
         if (outputPathStr == NULL) abort();

         for (int i=0; i<icComm->numCommBatches(); i++) {
            int chars_needed = snprintf(outputPathStr, len, "%s/batchsweep_%0*d/", outputPathName, lenserialno, i);
            assert(chars_needed < len);
            activeBatchSweep->pushStringValue(outputPathStr);
         }
         free(outputPathStr); outputPathStr = NULL;
         addActiveBatchSweep(hypercolgroupname, "outputPath");
      }

      if(!hasSweepValue("checkpointWriteDir")){
         const char * hypercolgroupname = NULL;
         const char * checkpointWriteDir = NULL;
         for (int g=0; g<numGroups; g++) {
            if (groups[g]->getGroupKeyword(),"HyPerCol") {
               hypercolgroupname = groups[g]->name();
               checkpointWriteDir = groups[g]->stringValue("checkpointWriteDir");
               break;
            }
         }
         if (hypercolgroupname == NULL) {
            fprintf(stderr, "PVParams::parseBuffer error: no HyPerCol group\n");
            abort();
         }
         if(checkpointWriteDir){
            char dummy;
            int lenserialno = snprintf(&dummy, 0, "%d", batchSweepSize-1);
            int len = snprintf(&dummy, 0, "%s/batchsweep_%0*d/", checkpointWriteDir, lenserialno, parameterSweepSize-1)+1;
            char * checkpointPathStr = (char *) calloc(len, sizeof(char));
            if (checkpointPathStr == NULL) abort();
            for (int i=0; i<icComm->numCommBatches(); i++) {
               int chars_needed = snprintf(checkpointPathStr, len, "%s/batchsweep_%0*d/", checkpointWriteDir, lenserialno, i);
               assert(chars_needed < len);
               activeBatchSweep->pushStringValue(checkpointPathStr);
            }
            free(checkpointPathStr); checkpointPathStr= NULL;
            addActiveBatchSweep(hypercolgroupname, "checkpointWriteDir");
         }
      }
   }

   // Each ParameterSweep needs to have its group/parameter pair added to the database, if it's not already present.
   for (int k=0; k<numberOfParameterSweeps(); k++) {
      ParameterSweep * sweep = paramSweeps[k];
      const char * group_name = sweep->getGroupName();
      const char * param_name = sweep->getParamName();
      SweepType type = sweep->getType();
      ParameterGroup * g = group(group_name);
      if (g==NULL) {
         fprintf(stderr, "ParameterSweep error: there is no group \"%s\"\n", group_name);
         abort();
      }
      switch (type) {
      case SWEEP_NUMBER:
         if (!g->present(param_name) ) {
            Parameter * p = new Parameter(param_name, 0.0);
            g->pushNumerical(p);
         }
         break;
      case SWEEP_STRING:
         if (!g->stringPresent(param_name)) {
            ParameterString * p = new ParameterString(param_name, "");
            g->pushString(p);
         }
         break;
      default:
         assert(0);
         break;
      }
   }

   for (int k=0; k<numberOfBatchSweeps(); k++) {
      ParameterSweep * sweep = batchSweeps[k];
      const char * group_name = sweep->getGroupName();
      const char * param_name = sweep->getParamName();
      SweepType type = sweep->getType();
      ParameterGroup * g = group(group_name);
      if (g==NULL) {
         fprintf(stderr, "BatchSweep error: there is no group \"%s\"\n", group_name);
         abort();
      }
      switch (type) {
      case SWEEP_NUMBER:
         if (!g->present(param_name) ) {
            Parameter * p = new Parameter(param_name, 0.0);
            g->pushNumerical(p);
         }
         break;
      case SWEEP_STRING:
         if (!g->stringPresent(param_name)) {
            ParameterString * p = new ParameterString(param_name, "");
            g->pushString(p);
         }
         break;
      default:
         assert(0);
         break;
      }
   }

   clearHasBeenReadFlags();

   return PV_SUCCESS;
}

#ifdef OBSOLETE // Marked obsolete Aug 30, 2015. Never gets called anywhere in the OpenPV repository, and undocumented.
int PVParams::parseBufferInRootProcess(char * buffer, long int bufferLength) {
   // Under MPI, if this process is called, it should be called by all processes.
#ifdef PV_USE_MPI
   int status = PV_SUCCESS;
   if (worldRank==0) {
      MPI_Bcast(&bufferLength, 1, MPI_LONG, 0, icComm->globalCommunicator());
      MPI_Bcast(buffer, bufferLength, MPI_CHAR, 0, icComm->globalCommunicator());
      status = parseBuffer(buffer, bufferLength);
   }
   else {
      long int bufLen;
      char * buf = NULL;
      MPI_Bcast(&bufLen, 1, MPI_LONG, 0, icComm->globalCommunicator());
      buf = (char *) malloc((size_t) bufLen);
      if (buf==NULL) {
         fprintf(stderr, "Process %d: error allocating %ld bytes for PVParams buffer.\n", worldRank, bufLen);
         exit(EXIT_FAILURE);
      }
      MPI_Bcast(buf, bufLen, MPI_CHAR, 0, icComm->globalCommunicator());
      status = parseBuffer(buf, bufLen);
      free(buf);
   }
#else // PV_USE_MPI
   status = parseBuffer(buffer, bufferLength);
#endif // PV_USE_MPI
   return status;
}
#endif // OBSOLETE // Marked obsolete Aug 30, 2015. Never gets called anywhere in the OpenPV repository, and undocumented.

int PVParams::setBatchSweepSize() {
   //std::cout << "Exiting test\n";
   batchSweepSize = -1;
   for (int k=0; k<numBatchSweeps; k++) {
      if (batchSweepSize<0) {
         batchSweepSize = this->batchSweeps[k]->getNumValues();
      }
      else {
         if (batchSweepSize != this->batchSweeps[k]->getNumValues()) {
            fprintf(stderr, "PVParams::setBatchSweepSize error: all BatchSweeps in the parameters file must have the same number of entries.\n");
            abort();
         }
      }
   }
   if (batchSweepSize < 0) batchSweepSize = 0;
   int batchWidth = icComm->numCommBatches();
   if(batchSweepSize){
      if(batchWidth != batchSweepSize){
         fprintf(stderr, "PVParams::setBatchSweepSize error: batchSweepSize %d must be the same as the MPI batch width %d.\n", batchSweepSize, batchWidth);
         exit(-1);
      }
   }
   return batchSweepSize;
}

int PVParams::setParameterSweepSize() {
   parameterSweepSize = -1;
   for (int k=0; k<this->numberOfParameterSweeps(); k++) {
      if (parameterSweepSize<0) {
         parameterSweepSize = this->paramSweeps[k]->getNumValues();
      }
      else {
         if (parameterSweepSize != this->paramSweeps[k]->getNumValues()) {
            fprintf(stderr, "PVParams::setParameterSweepSize error: all ParameterSweeps in the parameters file must have the same number of entries.\n");
            abort();
         }
      }
   }
   if (parameterSweepSize < 0) parameterSweepSize = 0;
   return parameterSweepSize;
}

int PVParams::setBatchSweepValues(){
   int status = PV_SUCCESS;
   //Set batch sweeps
   //Use communicator to determine which values to use
   int batchRank = icComm->commBatch();
   for (int k=0; k<numBatchSweeps; k++) {
      ParameterSweep * batchSweep = batchSweeps[k];
      SweepType type = batchSweep->getType();
      const char * group_name = batchSweep->getGroupName();
      const char * param_name = batchSweep->getParamName();
      ParameterGroup * gp = group(group_name);
      assert(gp!=NULL);
      const char * s;
      double v = 0.0f;
      switch (type) {
      case SWEEP_NUMBER:
         batchSweep->getNumericValue(batchRank, &v);
         gp->setValue(param_name, v);
         break;
      case SWEEP_STRING:
         s = batchSweep->getStringValue(batchRank);
         gp->setStringValue(param_name, s);
         break;
      default:
         assert(0);
         break;
      }
   }
   return status;
}

int PVParams::setParameterSweepValues(int n) {
   int status = PV_SUCCESS;
   //Set parameter sweeps
   if (n<0 || n>=parameterSweepSize) {
      status = PV_FAILURE;
      return status;
   }
   for (int k=0; k<this->numberOfParameterSweeps(); k++) {
      ParameterSweep * paramSweep = paramSweeps[k];
      SweepType type = paramSweep->getType();
      const char * group_name = paramSweep->getGroupName();
      const char * param_name = paramSweep->getParamName();
      ParameterGroup * gp = group(group_name);
      assert(gp!=NULL);

      const char * s;
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
      default:
         assert(0);
         break;
      }
   }
   return status;
}

/**
 * @groupName
 * @paramName
 */
int PVParams::present(const char * groupName, const char * paramName)
{
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      if( worldRank == 0 ) {
         fprintf(stderr, "PVParams::present: ERROR, couldn't find a group for %s\n",
                 groupName);
      }
      exit(EXIT_FAILURE);
   }

   return g->present(paramName);
}

/**
 * @groupName
 * @paramName
 */
double PVParams::value(const char * groupName, const char * paramName)
{
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      if( worldRank == 0 ) {
         fprintf(stderr, "PVParams::value: ERROR, couldn't find a group for %s\n",
                 groupName);
      }
      exit(EXIT_FAILURE);
   }

   return g->value(paramName);
}

int PVParams::valueInt(const char * groupName, const char * paramName)
{
   double v = value(groupName, paramName);
   return convertParamToInt(v);
}

int PVParams::valueInt(const char * groupName, const char * paramName, int initialValue, bool warnIfAbsent)
{
   double v = value(groupName, paramName, (double) initialValue, warnIfAbsent);
   return convertParamToInt(v);
}

int PVParams::convertParamToInt(double value)
{
   int y=0;
   if (value>=(double)INT_MAX) { y = INT_MAX;}
   else if (value<=(double)INT_MIN) { y = INT_MIN; }
   else { y = (int) nearbyint(value); }
   return y;
}

/**
 * @groupName
 * @paramName
 * @initialValue
 */
double PVParams::value(const char * groupName, const char * paramName, double initialValue, bool warnIfAbsent)
{
   if (present(groupName, paramName)) {
      return value(groupName, paramName);
   }
   else {
      if( warnIfAbsent && worldRank == 0 ) {
          printf("Using default value %f for parameter \"%s\" in group \"%s\"\n", initialValue, paramName, groupName);
      }
      return initialValue;
   }
}

bool PVParams::arrayPresent(const char * groupName, const char * paramName) {
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      if( worldRank == 0 ) {
         fprintf(stderr, "PVParams::present: ERROR, couldn't find a group for %s\n",
                 groupName);
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
const float * PVParams::arrayValues(const char * groupName, const char * paramName, int * size, bool warnIfAbsent) {
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      if( worldRank == 0 ) {
         fprintf(stderr, "PVParams::value: ERROR, couldn't find a group for %s\n",
               groupName);
      }
      return NULL;
   }
   const float * retval = g->arrayValues(paramName, size);
   if (retval == NULL) {
      assert(*size==0);
      if (worldRank==0) {
         fprintf(stderr, "Using empty array for parameter \"%s\" in group \"%s\"\n", paramName, groupName);
      }
   }
   return retval;
}

/*
 *  @groupName
 *  @paramName
 *  @size
 */
const double * PVParams::arrayValuesDbl(const char * groupName, const char * paramName, int * size, bool warnIfAbsent) {
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      if( worldRank == 0 ) {
         fprintf(stderr, "PVParams::value: ERROR, couldn't find a group for %s\n",
               groupName);
      }
      return NULL;
   }
   const double * retval = g->arrayValuesDbl(paramName, size);
   if (retval == NULL) {
      assert(*size==0);
      if (worldRank==0) {
         fprintf(stderr, "Using empty array for parameter \"%s\" in group \"%s\"\n", paramName, groupName);
      }
   }
   return retval;
}

/*
 *  @groupName
 *  @paramStringName
 */
int PVParams::stringPresent(const char * groupName, const char * paramStringName) {
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      if( worldRank == 0 ) {
         fprintf(stderr, "PVParams::value: ERROR, couldn't find a group for %s\n",
                 groupName);
      }
      exit(EXIT_FAILURE);
   }

   return g->stringPresent(paramStringName);
}

/*
 *  @groupName
 *  @paramStringName
 */
const char * PVParams::stringValue(const char * groupName, const char * paramStringName, bool warnIfAbsent) {
   if( stringPresent(groupName, paramStringName) ) {
      ParameterGroup * g = group(groupName);
      return g->stringValue(paramStringName);
   }
   else {
      if( warnIfAbsent && worldRank == 0 ) {
         printf("No parameter string named \"%s\" in group \"%s\"\n", paramStringName, groupName);
      }
      return NULL;
   }
}

/**
 * @groupName
 */
ParameterGroup * PVParams::group(const char * groupName)
{
   for (int i = 0; i < numGroups; i++) {
      if (strcmp(groupName, groups[i]->name()) == 0) {
         return groups[i];
      }
   }
   return NULL;
}

const char * PVParams::groupNameFromIndex(int index) {
   bool inbounds = index >= 0 && index < numGroups;
   return inbounds ? groups[index]->name() : NULL;
}

const char * PVParams::groupKeywordFromIndex(int index) {
   bool inbounds = index >= 0 && index < numGroups;
   return inbounds ? groups[index]->getGroupKeyword() : NULL;
}

const char * PVParams::groupKeywordFromName(const char * name) {
   const char * kw = NULL;
   ParameterGroup * g = group(name);
   if (g!=NULL) {
      kw = g->getGroupKeyword();
   }
   return kw;
}

/**
 * @keyword
 * @name
 */
void PVParams::addGroup(char * keyword, char * name)
{
   assert((size_t) numGroups <= groupArraySize);

   // Verify that the new group's name is not an existing group's name
   for( int k=0; k<numGroups; k++ ) {
      if( !strcmp(name, groups[k]->name())) {
         fprintf(stderr, "Rank %d process: group name \"%s\" duplicated\n", worldRank, name);
         exit(EXIT_FAILURE);
      }
   }

   if( (size_t) numGroups == groupArraySize ) {
      groupArraySize += RESIZE_ARRAY_INCR;
      ParameterGroup ** newGroups = (ParameterGroup **) malloc( groupArraySize * sizeof(ParameterGroup *) );
      assert(newGroups);
      for(  int k=0; k< numGroups; k++ ) {
         newGroups[k] = groups[k];
      }
      free(groups);
      groups = newGroups;
   }

   groups[numGroups] = new ParameterGroup(name, stack, arrayStack, stringStack, worldRank);
   groups[numGroups]->setGroupKeyword(keyword);

   // the parameter group takes over control of the PVParams's stack and stringStack; make new ones.
   stack = new ParameterStack(MAX_PARAMS);
   arrayStack = new ParameterArrayStack(PARAMETERARRAYSTACK_INITIALCOUNT);
   stringStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);

   numGroups++;
}

void PVParams::addActiveParamSweep(const char * group_name, const char * param_name) {
   //Search for group_name and param_name in both ParameterSweep and BatchSweep list of objects
   for(int p = 0; p < numParamSweeps; p++){
      if(strcmp(paramSweeps[p]->getGroupName(), group_name) == 0 && strcmp(paramSweeps[p]->getParamName(), param_name) == 0){
         fprintf(stderr, "PVParams::addActiveParamSweep: Parameter sweep %s, %s already exists\n", group_name, param_name);
         abort();
      }
   }
   for(int b = 0; b < numBatchSweeps; b++){
      if(strcmp(batchSweeps[b]->getGroupName(), group_name) == 0 && strcmp(batchSweeps[b]->getParamName(), param_name) == 0){
         fprintf(stderr, "PVParams::addActiveParamSweep: Parameter sweep %s, %s already exists as a batch sweep, cannot do both for same parameter.\n", group_name, param_name);
         abort();
      }
   }

   activeParamSweep->setGroupAndParameter(group_name, param_name);
   ParameterSweep ** newParamSweeps = (ParameterSweep **) calloc(numParamSweeps+1, sizeof(ParameterSweep *));
   if (newParamSweeps == NULL) {
      fprintf(stderr, "PVParams::action_parameter_sweep: unable to allocate memory for larger paramSweeps\n");
      abort();
   }
   for (int k=0; k<numParamSweeps; k++) {
      newParamSweeps[k] = paramSweeps[k];
   }
   free(paramSweeps);
   paramSweeps = newParamSweeps;
   paramSweeps[numParamSweeps] = activeParamSweep;
   numParamSweeps++;
   newActiveParamSweep();
}

void PVParams::addActiveBatchSweep(const char * group_name, const char * param_name) {
   //Search for group_name and param_name in both ParameterSweep and BatchSweep list of objects
   for(int b = 0; b < numBatchSweeps; b++){
      if(strcmp(batchSweeps[b]->getGroupName(), group_name) == 0 && strcmp(batchSweeps[b]->getParamName(), param_name) == 0){
         fprintf(stderr, "PVParams::addActiveBatchSweep: Batch sweep %s, %s already exists\n", group_name, param_name);
         abort();
      }
   }
   for(int p = 0; p < numParamSweeps; p++){
      if(strcmp(paramSweeps[p]->getGroupName(), group_name) == 0 && strcmp(paramSweeps[p]->getParamName(), param_name) == 0){
         fprintf(stderr, "PVParams::addActiveParamSweep: Batch sweep %s, %s already exists as a parameter sweep, cannot do both for same parameter.\n", group_name, param_name);
         abort();
      }
   }

   activeBatchSweep->setGroupAndParameter(group_name, param_name);
   ParameterSweep ** newBatchSweeps = (ParameterSweep **) calloc(numBatchSweeps+1, sizeof(ParameterSweep *));
   if (newBatchSweeps == NULL) {
      fprintf(stderr, "PVParams::action_parameter_sweep: unable to allocate memory for larger batchSweeps\n");
      abort();
   }
   for (int k=0; k<numBatchSweeps; k++) {
      newBatchSweeps[k] = batchSweeps[k];
   }
   free(batchSweeps);
   batchSweeps = newBatchSweeps;
   batchSweeps[numBatchSweeps] = activeBatchSweep;
   numBatchSweeps++;
   newActiveBatchSweep();
}

int PVParams::warnUnread() {
   int status = PV_SUCCESS;
   for( int i=0; i<numberOfGroups(); i++) {
      if( groups[i]->warnUnread() != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

bool PVParams::hasBeenRead(const char * group_name, const char * param_name) {
   ParameterGroup * g = group(group_name);
   if (g == NULL) {
      return false;
   }

   return g->hasBeenRead(param_name);
}

bool PVParams::presentAndNotBeenRead(const char * group_name, const char * param_name) {
   bool is_present = present(group_name, param_name);
   if (!is_present) is_present = arrayPresent(group_name, param_name);
   if (!is_present) is_present = stringPresent(group_name, param_name);
   bool has_been_read = hasBeenRead(group_name, param_name);
   return is_present && !has_been_read;
}

int PVParams::clearHasBeenReadFlags() {
   int status = PV_SUCCESS;
   for (int i=0; i<numberOfGroups(); i++) {
      if (groups[i]->clearHasBeenReadFlags() != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   return status;
}

void PVParams::handleUnnecessaryParameter(const char * group_name, const char * param_name) {
   if (present(group_name, param_name)) {
      if (worldRank==0) {
         const char * class_name = groupKeywordFromName(group_name);
         fprintf(stderr, "Warning: %s \"%s\" does not use parameter %s, but it is present in the parameters file.\n",
               class_name, group_name, param_name);
      }
      value(group_name, param_name); // marks param as read so that presentAndNotBeenRead doesn't trip up
   }
}

template <typename T>
void PVParams::handleUnnecessaryParameter(const char * group_name, const char * param_name, T correct_value) {
   int status = PV_SUCCESS;
   if (present(group_name, param_name)) {
      if (worldRank==0) {
         const char * class_name = groupKeywordFromName(group_name);
         fprintf(stderr, "Warning: %s \"%s\" does not use parameter %s, but it is present in the parameters file.\n",
               group_name, class_name, param_name);
      }
      T params_value = (T) value(group_name, param_name); // marks param as read so that presentAndNotBeenRead doesn't trip up
      if (params_value != correct_value) {
         status = PV_FAILURE;
         if (worldRank==0) {
            std::cerr << "   Value " << params_value << " is inconsistent with correct value " << correct_value;
            std::cerr << std::endl; // This line is separate from above line to avoid an Eclipse bug that flags a nonexistent invalid overload
         }
      }
   }
   MPI_Barrier(icComm->globalCommunicator());
   if (status != PV_SUCCESS) exit(EXIT_FAILURE);
}
// Declare the instantiations of allocateBuffer that occur in other .cpp files; otherwise you may get linker errors.
template void PVParams::handleUnnecessaryParameter<bool>(const char * group_name, const char * param_name, bool correct_value);
template void PVParams::handleUnnecessaryParameter<int>(const char * group_name, const char * param_name, int correct_value);
template void PVParams::handleUnnecessaryParameter<float>(const char * group_name, const char * param_name, float correct_value);
template void PVParams::handleUnnecessaryParameter<double>(const char * group_name, const char * param_name, double correct_value);

void PVParams::handleUnnecessaryStringParameter(const char * group_name, const char * param_name) {
   int status = PV_SUCCESS;
   const char * class_name = groupKeywordFromName(group_name);
   if (stringPresent(group_name, param_name)) {
      if (worldRank==0) {
         fprintf(stderr, "Warning: %s \"%s\" does not use string parameter %s, but it is present in the parameters file.\n",
               class_name, group_name, param_name);
      }
      const char * params_value = stringValue(group_name, param_name, false/*warnIfAbsent*/); // marks param as read so that presentAndNotBeenRead doesn't trip up
      assert(params_value);
   }
   const char * params_value = stringValue(group_name, param_name, false/*warnIfAbsent*/); // marks param as read so that presentAndNotBeenRead doesn't trip up
}
void PVParams::handleUnnecessaryStringParameter(const char * group_name, const char * param_name, const char * correct_value, bool case_insensitive_flag) {
   int status = PV_SUCCESS;
   const char * class_name = groupKeywordFromName(group_name);
   if (stringPresent(group_name, param_name)) {
      if (worldRank==0) {
         fprintf(stderr, "Warning: %s \"%s\" does not use string parameter %s, but it is present in the parameters file.\n",
               class_name, group_name, param_name);
      }
      const char * params_value = stringValue(group_name, param_name, false/*warnIfAbsent*/); // marks param as read so that presentAndNotBeenRead doesn't trip up
      if (params_value != NULL && correct_value != NULL) {
         char * correct_value_i = strdup(correct_value); // need mutable strings for case-insensitive comparison
         char * params_value_i = strdup(params_value); // need mutable strings for case-insensitive comparison
         if (correct_value_i == NULL) {
            status = PV_FAILURE;
            if (worldRank==0) {
               fprintf(stderr, "%s \"%s\" error: Rank %d process unable to copy correct string value: %s.\n",
                     class_name, group_name, worldRank, strerror(errno));
            }
         }
         if (params_value_i == NULL) {
            status = PV_FAILURE;
            if (worldRank==0) {
               fprintf(stderr, "%s \"%s\" error: Rank %d process unable to copy parameter string value: %s.\n",
                     class_name, group_name, worldRank, strerror(errno));
            }
         }
         if (case_insensitive_flag) {
            for (char * c = params_value_i; *c!='\0'; c++) {
               *c = (char) tolower((int) *c);
            }
            for (char * c = correct_value_i; *c!='\0'; c++) {
               *c = (char) tolower((int) *c);
            }
         }
         if (strcmp(params_value_i, correct_value_i) != 0) {
            status = PV_FAILURE;
            if (worldRank==0) {
               fprintf(stderr, "%s \"%s\" error: parameter string %s = \"%s\" is inconsistent with correct value \"%s\".  Exiting.\n",
                     class_name, group_name, param_name, params_value, correct_value);
            }
         }
         free(correct_value_i);
         free(params_value_i);
      }
      else if (params_value == NULL && correct_value != NULL) {
         status = PV_FAILURE;
         if (worldRank==0) {
            fprintf(stderr, "%s \"%s\" error: parameter string %s = NULL is inconsistent with correct value \"%s\".  Exiting.\n",
                  class_name, group_name, param_name, correct_value);
         }
      }
      else if (params_value != NULL && correct_value == NULL) {
         status = PV_FAILURE;
         if (worldRank==0) {
            fprintf(stderr, "%s \"%s\" error: parameter string %s = \"%s\" is inconsistent with correct value of NULL.  Exiting.\n",
                  class_name, group_name, param_name, params_value);
         }
      }
      else {
         assert(params_value==NULL && correct_value == NULL);
         assert(status==PV_SUCCESS);
      }
   }
   if (status != PV_SUCCESS) {
      MPI_Barrier(icComm->globalCommunicator());
      exit(EXIT_FAILURE);
   }
}

int PVParams::outputParams(FILE * fp) {
   int status = PV_SUCCESS;
   for( int g=0; g<numGroups; g++ ) {
      if( groups[g]->outputGroup(fp)!=PV_SUCCESS ) status = PV_FAILURE;
   }
   return status;
}

/**
 * @id
 * @val
 */
void PVParams::action_pvparams_directive(char * id, double val)
{
   if( !strcmp(id,"debugParsing") ) {
      debugParsing = (val != 0);
      if( worldRank == 0 ) {
         printf("debugParsing turned ");
         if(debugParsing) {
            printf("on.\n");
         }
         else {
            printf("off.\n");
         }
      }
   }
   else if ( !strcmp(id, "disable") ) {
      disable = (val != 0);
      if (worldRank == 0 ) {
         printf("Parsing params file ");
         if (disable) {
            printf("disabled.\n");
         }
         else {
            printf("enabled.\n");
         }
      }
   }
   else {
      if (worldRank == 0) {
         fprintf(stderr,"Unrecognized directive %s = %f, skipping.\n", id, val);
      }
   }
}

/**
 * @keyword
 * @name
 */
void PVParams::action_parameter_group()
{
   if (disable) return;
   if(debugParsing && worldRank==0 ) {
      printf("action_parameter_group: %s \"%s\" parsed successfully.\n", currGroupKeyword, currGroupName);
      fflush(stdout);
   }
   // build a parameter group
   addGroup(currGroupKeyword, currGroupName);
}
void PVParams::action_parameter_group_name(char * keyword, char * name)
{
   if (disable) return;
   // remove surrounding quotes
   int len = strlen(++name);
   name[len-1] = '\0';

   if(debugParsing && worldRank==0 ) {
      printf("action_parameter_group_name: %s \"%s\" parsed successfully.\n", keyword, name);
      fflush(stdout);
   }
   currGroupKeyword = keyword;
   currGroupName = name;
}

/**
 * @id
 * @val
 */
void PVParams::action_parameter_def(char * id, double val)
{
   if (disable) return;
   if(debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_parameter_def: %s = %lf\n", id, val);
      fflush(stdout);
   }
   if( checkDuplicates(id, val) != PV_SUCCESS ) exit(EXIT_FAILURE);
   Parameter * p = new Parameter(id, val);
   stack->push(p);
}

void PVParams::action_parameter_def_overwrite(char * id, double val){
   if (disable) return;
   if (debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_parameter_def_overwrite: %s = %lf\n", id, val);
      fflush(stdout);
   }
   //Search through current parameters for the id
   char * param_name = stripOverwriteTag(id);
   Parameter* currParam = NULL;
   for (int i = 0; i < stack->size(); i++){
      Parameter* param = stack->peek(i);
      if(strcmp(param->name(), param_name) == 0){
         currParam = param;
      }
   }
   if(!currParam){
      for (int i = 0; i < arrayStack->size(); i++){
         ParameterArray* arrayParam = arrayStack->peek(i);
         if(strcmp(arrayParam->name(), param_name) == 0){
            fflush(stdout);
            printf("%s is defined as an array parameter. Overwriting array parameters with value parameters not implemented yet.\n", id);
            fflush(stdout);
            exit(EXIT_FAILURE);
         }
      }
      fflush(stdout);
      printf("Overwrite error: %s is not an existing parameter to overwrite.\n", id);
      fflush(stdout);
      exit(EXIT_FAILURE);
   }
   free(param_name);
   //Set to new value
   currParam->setValue(val);
}

void PVParams::action_parameter_array(char * id)
{
   if (disable) return;
   if (debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_parameter_array: %s\n", id);
      fflush(stdout);
   }
   int status = currentParamArray->setName(id);
   assert(status==PV_SUCCESS);
   if( checkDuplicates(id, 0.0) != PV_SUCCESS ) exit(EXIT_FAILURE);
   arrayStack->push(currentParamArray);
   currentParamArray = new ParameterArray(PARAMETERARRAYSTACK_INITIALCOUNT);
}

void PVParams::action_parameter_array_overwrite(char * id){
   if (disable) return;
   if (debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_parameter_array_overwrite: %s\n", id);
      fflush(stdout);
   }
   //Search through current parameters for the id
   char * param_name = stripOverwriteTag(id);
   ParameterArray * origArray = NULL;
   for (int i = 0; i < arrayStack->size(); i++){
      ParameterArray* arrayParam = arrayStack->peek(i);
      if(strcmp(arrayParam->name(), param_name) == 0){
         origArray = arrayParam;
      }
   }
   if(!origArray){
      for (int i = 0; i < stack->size(); i++){
         Parameter* param = stack->peek(i);
         if(strcmp(param->name(), param_name) == 0){
            fflush(stdout);
            printf("%s is defined as a value parameter. Overwriting value parameters with array parameters not implemented yet.\n", id);
            fflush(stdout);
            exit(EXIT_FAILURE);
         }
      }
      fflush(stdout);
      printf("Overwrite error: %s is not an existing parameter to overwrite.\n", id);
      fflush(stdout);
      exit(EXIT_FAILURE);
   }
   free(param_name);
   //Set values of arrays
   origArray->resetArraySize();
   for(int i = 0; i < currentParamArray->getArraySize(); i++){ 
      origArray->pushValue(currentParamArray->peek(i));
   }
   assert(origArray->getArraySize() == currentParamArray->getArraySize());
   delete currentParamArray;
   currentParamArray = new ParameterArray(PARAMETERARRAYSTACK_INITIALCOUNT);
}

void PVParams::action_parameter_array_value(double val)
{
   if (disable) return;
   if(debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_parameter_array_value %lf\n", val);
   }
   int sz = currentParamArray->getArraySize();
   int newsize = currentParamArray->pushValue(val);
   assert(newsize == sz+1);
}

void PVParams::action_parameter_string_def(const char * id, const char * stringval) {
   if (disable) return;
   if( debugParsing && worldRank == 0 ) {
      fflush(stdout);
      printf("action_parameter_string_def: %s = %s\n", id, stringval);
      fflush(stdout);
   }
   if( checkDuplicates(id, 0.0) != PV_SUCCESS ) exit(EXIT_FAILURE);
   char * param_value = stripQuotationMarks(stringval);
   assert(!stringval || param_value); // stringval can be null, but if stringval is not null, param_value should also be non-null
   ParameterString * pstr = new ParameterString(id, param_value);
   stringStack->push(pstr);
   free(param_value);
}

void PVParams::action_parameter_string_def_overwrite(const char * id, const char * stringval){
   if (disable) return;
   if( debugParsing && worldRank == 0 ) {
      fflush(stdout);
      printf("action_parameter_string_def_overwrite: %s = %s\n", id, stringval);
      fflush(stdout);
   }
   //Search through current parameters for the id
   char * param_name = stripOverwriteTag(id);
   ParameterString* currParam = NULL;
   for (int i = 0; i < stringStack->size(); i++){
      ParameterString* param = stringStack->peek(i);
      assert(param);
      if(strcmp(param->getName(), param_name) == 0){
         currParam = param;
      }
   }
   free(param_name);
   if(!currParam){
      fflush(stdout);
      printf("Overwrite error: %s is not an existing parameter to overwrite.\n", id);
      fflush(stdout);
      exit(EXIT_FAILURE);
   }
   char * param_value = stripQuotationMarks(stringval);
   assert(!stringval || param_value); // stringval can be null, but if stringval is not null, param_value should also be non-null
   //Set to new value
   currParam->setValue(param_value);
   free(param_value);
}

void PVParams::action_parameter_filename_def(const char * id, const char * stringval) {
   if (disable) return;
   if( debugParsing && worldRank == 0 ) {
      fflush(stdout);
      printf("action_parameter_filename_def: %s = %s\n", id, stringval);
      fflush(stdout);
   }
   if( checkDuplicates(id, 0.0) != PV_SUCCESS ) { exit(EXIT_FAILURE); }
   char * param_value = stripQuotationMarks(stringval);
   assert(param_value);
   ParameterString * pstr = NULL;
   char * filename = NULL;
   if (param_value && param_value[0]=='~') {
      filename = expandLeadingTilde(param_value);
      pstr = new ParameterString(id, filename);
      free(filename);
   }
   else {
      pstr = new ParameterString(id, param_value);
   }
   free(param_value);
   stringStack->push(pstr);
}

void PVParams::action_parameter_filename_def_overwrite(const char * id, const char * stringval){
   if (disable) return;
   if( debugParsing && worldRank == 0 ) {
      fflush(stdout);
      printf("action_parameter_filename_def_overwrite: %s = %s\n", id, stringval);
      fflush(stdout);
   }
   //Search through current parameters for the id
   char * param_name = stripOverwriteTag(id);
   ParameterString* currParam = NULL;
   for (int i = 0; i < stringStack->size(); i++){
      ParameterString* param = stringStack->peek(i);
      assert(param);
      if(strcmp(param->getName(), param_name) == 0){
         currParam = param;
      }
   }
   free(param_name); param_name = NULL;
   if(!currParam){
      fflush(stdout);
      printf("Overwrite error: %s is not an existing parameter to overwrite.\n", id);
      fflush(stdout);
      exit(EXIT_FAILURE);
   }
   char * param_value = stripQuotationMarks(stringval);
   assert(param_value);
   char * filename = NULL;
   if (param_value && param_value[0]=='~') {
      filename = expandLeadingTilde(param_value);
      currParam->setValue(filename);
   }
   else {
      currParam->setValue(param_value);
   }
   free(param_value);
}

void PVParams::action_include_directive(const char * stringval) {
   if (disable) return;
   if( debugParsing && worldRank == 0 ) {
      fflush(stdout);
      printf("action_include_directive: including %s\n", stringval);
      fflush(stdout);
   }
   //The include directive must be the first parameter in the group if defined
   if(stack->size() != 0 || arrayStack->size() != 0 || stringStack->size() != 0){
      fflush(stdout);
      printf("Import of %s must be the first parameter specified in the group.\n", stringval);
      fflush(stdout);
      exit(EXIT_FAILURE);
   }
   //Grab the parameter value
   char * param_value = stripQuotationMarks(stringval);
   //Grab the included group's ParameterGroup object
   ParameterGroup * includeGroup = NULL;
   for(int groupidx = 0; groupidx < numGroups; groupidx++){
      //If strings are matching
      if(strcmp(groups[groupidx]->name(), param_value) == 0){
         includeGroup = groups[groupidx];
      }
   }
   //If group not found
   if(!includeGroup){
      fflush(stdout);
      printf("Include error: include group %s is not defined.\n", param_value);
      fflush(stdout);
      exit(EXIT_FAILURE);
   }
   //Check keyword of group
   if(strcmp(includeGroup->getGroupKeyword(), currGroupKeyword) != 0){
      fflush(stdout);
      printf("Include error: Cannot include group %s, which is a %s, into a %s. Group types must be the same.\n", param_value, includeGroup->getGroupKeyword(), currGroupKeyword);
      fflush(stdout);
      exit(EXIT_FAILURE);
   }
   free(param_value);
   //Load all stack values into current parameter group

   assert(stack->size()==0);
   delete stack;
   stack = includeGroup->copyStack();

   assert(arrayStack->size()==0);
   delete arrayStack;
   arrayStack = includeGroup->copyArrayStack();

   assert(stringStack->size()==0);
   delete stringStack;
   stringStack = includeGroup->copyStringStack();
}

void PVParams::action_sweep_open(const char * groupname, const char * paramname)
{
   if (disable) return;
   // strip quotation marks from groupname
   currSweepGroupName = stripQuotationMarks(groupname);
   assert(currSweepGroupName);
   currSweepParamName = strdup(paramname);
   if (debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_batch_sweep_open: Sweep for group %s, parameter \"%s\" starting\n", groupname, paramname);
      fflush(stdout);
   }
}

void PVParams::action_parameter_sweep_close()
{
   if (disable) return;
   addActiveParamSweep(currSweepGroupName, currSweepParamName);
   if(debugParsing && worldRank==0 ) {
      printf("action_parameter_group: ParameterSweep for %s \"%s\" parsed successfully.\n", currSweepGroupName, currSweepParamName);
      fflush(stdout);
   }
   // build a parameter group
   free(currSweepGroupName);
   free(currSweepParamName);
}

void PVParams::action_batch_sweep_close()
{
   if (disable) return;
   addActiveBatchSweep(currSweepGroupName, currSweepParamName);
   if(debugParsing && worldRank==0 ) {
      printf("action_parameter_group: BatchSweep for %s \"%s\" parsed successfully.\n", currSweepGroupName, currSweepParamName);
      fflush(stdout);
   }
   // build a parameter group
   free(currSweepGroupName);
   free(currSweepParamName);
}

void PVParams::action_parameter_sweep_values_number(double val)
{
   if (disable) return;
   if (debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_parameter_sweep_values_number: %f\n", val);
      fflush(stdout);
   }
   activeParamSweep->pushNumericValue(val);
}

void PVParams::action_batch_sweep_values_number(double val)
{
   if (disable) return;
   if (debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_batch_sweep_values_number: %f\n", val);
      fflush(stdout);
   }
   activeBatchSweep->pushNumericValue(val);
}

void PVParams::action_parameter_sweep_values_string(const char * stringval)
{
   if (disable) return;
   if (debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_sweep_values_string: %s\n", stringval);
      fflush(stdout);
   }
   char * string = stripQuotationMarks(stringval);
   assert(!stringval || string); // stringval can be null, but if stringval is not null, string should also be non-null
   activeParamSweep->pushStringValue(string);
   free(string);
}

void PVParams::action_batch_sweep_values_string(const char * stringval)
{
   if (disable) return;
   if (debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_batch_values_string: %s\n", stringval);
      fflush(stdout);
   }
   char * string = stripQuotationMarks(stringval);
   assert(!stringval || string); // stringval can be null, but if stringval is not null, string should also be non-null
   activeBatchSweep->pushStringValue(string);
   free(string);
}

void PVParams::action_parameter_sweep_values_filename(const char * stringval)
{
   if (disable) return;
   if (debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_sweep_values_filename: %s\n", stringval);
      fflush(stdout);
   }
   char * filename = stripQuotationMarks(stringval);
   assert(filename);
   if (filename && filename[0]=='~') {
      char * newfilename = expandLeadingTilde(filename);
      free(filename);
      filename = newfilename;
   }
   activeParamSweep->pushStringValue(filename);
   free(filename);
}

void PVParams::action_batch_sweep_values_filename(const char * stringval)
{
   if (disable) return;
   if (debugParsing && worldRank == 0) {
      fflush(stdout);
      printf("action_batch_sweep_values_filename: %s\n", stringval);
      fflush(stdout);
   }
   char * filename = stripQuotationMarks(stringval);
   assert(filename);
   if (filename && filename[0]=='~') {
      char * newfilename = expandLeadingTilde(filename);
      free(filename);
      filename = newfilename;
   }
   activeBatchSweep->pushStringValue(filename);
   free(filename);
}

int PVParams::checkDuplicates(const char * paramName, double val) {
   int status = PV_SUCCESS;
   for( int k=0; k<stack->size(); k++ ) {
      Parameter * parm = stack->peek(k);
      if( !strcmp(paramName, parm->name() ) ) {
         double oldval = parm->value();
         if ( val == oldval) {
            fprintf(stderr, "Warning: parameter name \"%s\" duplicates a previous parameter name and value (%s = %f)\n", paramName, parm->name(), val);
         }
         else {
            fprintf(stderr, "Rank %d process: parameter name \"%s\" duplicates a previous parameter name with inconsistent values (%f versus %f)\n", worldRank, paramName, oldval, val);
            status = PV_FAILURE;
         }
         break;
      }
   }
   for( int k=0; k<arrayStack->size(); k++ ) {
      if( !strcmp(paramName, arrayStack->peek(k)->name() ) ) {
         fprintf(stderr, "Rank %d process: parameter name \"%s\" duplicates a previous array parameter name\n", worldRank, paramName);
         status = PV_FAILURE;
         break;
      }
   }
   for( int k=0; k<stringStack->size(); k++ ) {
      if( !strcmp(paramName, stringStack->peek(k)->getName() ) ) {
         fprintf(stderr, "Rank %d process: parameter name \"%s\" duplicates a previous string parameter name\n", worldRank, paramName);
         status = PV_FAILURE;
         break;
      }
   }
   if( status != PV_SUCCESS ) {
      if( numberOfGroups() == 0 ) {
         fprintf(stderr, "Rank %d process: this is the first parameter group being parsed\n", worldRank);
      }
      else {
         fprintf(stderr, "Rank %d process: last parameter group successfully added was \"%s\"\n", worldRank, groups[numberOfGroups()-1]->name());
      }
   }
   return status;
}

char * PVParams::stripQuotationMarks(const char * s) {
   // If a string has quotes as its first and last character, return the
   // part of the string inside the quotes, e.g. {'"', 'c', 'a', 't', '"'}
   // becomes {'c', 'a', 't'}.  If the string is null or does not have quotes at the
   // beginning and end, return NULL.
   // It is the responsibility of the routine that calls stripQuotationMarks
   // to free the returned string to avoid a memory leak.
   if (s==NULL) { return NULL;}
   char * noquotes = NULL;
   int len = strlen(s);
   if ( len >=2 && s[0]=='"' && s[len-1]=='"' ) {
      noquotes = (char *) calloc(len-1, sizeof(char));
      memcpy(noquotes, s+1, len-2);
      noquotes[len-2] = '\0'; // Not strictly necessary since noquotes was calloc'ed
   }
   return noquotes;
}

char * PVParams::stripOverwriteTag(const char * s){
   //Strips the @ tag to any overwritten params
   int len = strlen(s);
   char * notag = NULL;
   if(len >= 1 && s[0]=='@'){
      notag = (char *) calloc(len, sizeof(char));
      memcpy(notag, s+1, len-1);
      notag[len-1] = '\0';
   }
   return notag;
}

}  // close namespace PV block
