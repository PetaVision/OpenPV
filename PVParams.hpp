/*
 * PVParams.hpp
 *
 *  Created on: Nov 27, 2008
 *      Author: rasmussn
 */

#ifndef PVPARAMS_HPP_
#define PVPARAMS_HPP_

#include "../include/pv_common.h"
#include "../columns/HyPerCol.hpp"
#include "../columns/InterColComm.hpp"
#include <stdio.h>
#include <string.h>

// TODO - make MAX_PARAMS dynamic
#define MAX_PARAMS 100  // maximum number of parameters in a group

#undef HAS_MAIN   // define if provides a main function

namespace PV {

class InterColComm;

class Parameter {
public:
   Parameter(const char * name, double value);
   virtual ~Parameter();

   const char * name()      { return paramName; }
   double value()           { hasBeenReadFlag = true; return paramDblValue; }
   const float * valuePtr() { hasBeenReadFlag = true; return &paramValue; }
   const double * valueDblPtr() { hasBeenReadFlag = true; return &paramDblValue; }
   bool hasBeenRead()       { return hasBeenReadFlag; }
   int outputParam(FILE * fp, int indentation);
   void clearHasBeenRead()    { hasBeenReadFlag = false; }
   void setValue(double v)  { paramValue = (float) v; paramDblValue = v;}

private:
   char * paramName;
   float paramValue;
   double paramDblValue;
   bool   hasBeenReadFlag;
};

class ParameterArray {
public:
   ParameterArray(int initialSize);
   virtual ~ParameterArray();
   int getArraySize() {return arraySize;}
   const char * name() {return paramName;}
   int setName(const char * name);
   const float * getValues(int * sz) { hasBeenReadFlag = true; *sz = arraySize; return values;}
   const double * getValuesDbl(int * sz) { hasBeenReadFlag = true; *sz = arraySize; return valuesDbl;}
   int pushValue(double value);
   bool hasBeenRead() { return hasBeenReadFlag; }
   void clearHasBeenRead() { hasBeenReadFlag = false; }
   int outputString(FILE * fp, int indentation);

private:
   bool paramNameSet;
   char * paramName;
   int arraySize; // The number of values that have been pushed
   int bufferSize; // The size of the buffer in memory
   double * valuesDbl;
   float * values;
   bool hasBeenReadFlag;
};

class ParameterString {
public:
   ParameterString(const char * name, const char * value);
   virtual ~ParameterString();

   const char * getName()      { return paramName; }
   const char * getValue()     { hasBeenReadFlag = true; return paramValue; }
   bool hasBeenRead()          { return hasBeenReadFlag; }
   int outputString(FILE * fp, int indentation);
   void clearHasBeenRead()     { hasBeenReadFlag = false; }
   void setValue(const char * s) { free(paramValue); paramValue = strdup(s);}

private:
   char * paramName;
   char * paramValue;
   bool   hasBeenReadFlag;
};

class ParameterStack {
public:
   ParameterStack(int maxCount);
   virtual ~ParameterStack();

   int push(Parameter * param);
   Parameter * pop();
   Parameter * peek(int index)   { return parameters[index]; }
   int size()                    { return count; }
   int outputStack(FILE * fp, int indentation);

private:
   int count;
   int maxCount;
   Parameter ** parameters;
};

class ParameterArrayStack {
public:
   ParameterArrayStack(int initialCount);
   virtual ~ParameterArrayStack();
   int push(ParameterArray * array);
   int outputStack(FILE * fp, int indentation);
   int size() {return count;}
   ParameterArray * peek(int index) {return index>=0 && index<count ? parameterArrays[index] : NULL; }

private:
   int count; // Number of ParameterArrays
   int allocation; // Size of buffer
   ParameterArray ** parameterArrays;

};

class ParameterStringStack {
public:
   ParameterStringStack(int initialCount);
   virtual ~ParameterStringStack();

   int push(ParameterString * param);
   ParameterString * pop();
   ParameterString * peek(int index)    { return index>=0 && index<count ? parameterStrings[index] : NULL; }
   int size()                           { return count; }
   const char * lookup(const char * targetname);
   int outputStack(FILE * fp, int indentation);

private:
   int count;
   int allocation;
   ParameterString ** parameterStrings;
};

class ParameterGroup {
public:
   ParameterGroup(char * name, ParameterStack * stack, ParameterArrayStack * array_stack, ParameterStringStack * string_stack, int rank=0);
   virtual ~ParameterGroup();

   const char * name()   { return groupName; }
   const char * getGroupKeyword() { return groupKeyword; }
   int setGroupKeyword(const char * keyword);
   int setStringStack(ParameterStringStack * stringStack);
   int   present(const char * name);
   double value  (const char * name);
   bool  arrayPresent(const char * name);
   const float * arrayValues(const char * name, int * size);
   const double * arrayValuesDbl(const char * name, int * size);
   int   stringPresent(const char * stringName);
   const char * stringValue(const char * stringName);
   int warnUnread();
   bool hasBeenRead(const char * paramName);
   int clearHasBeenReadFlags();
   int outputGroup(FILE * fp);
   int pushNumerical(Parameter * param);
   int pushString(ParameterString * param);
   int setValue(const char * param_name, double value);
   int setStringValue(const char * param_name, const char * svalue);

private:
   char * groupName;
   char * groupKeyword;
   ParameterStack * stack;
   ParameterArrayStack * arrayStack;
   ParameterStringStack * stringStack;
   int processRank;
};


enum ParameterSweepType {
   PARAMSWEEP_UNDEF = 0,
   PARAMSWEEP_NUMBER  = 1,
   PARAMSWEEP_STRING  = 2
};

class ParameterSweep {
public:
   ParameterSweep();
   virtual ~ParameterSweep();

   int setGroupAndParameter(const char * groupname, const char * paramname);
   int pushNumericValue(double val);
   int pushStringValue(const char * sval);
   int getNumValues() {return numValues;}
   ParameterSweepType getType() {return type;}
   int getNumericValue(int n, double * val);
   const char * getStringValue(int n);
   const char * getGroupName() {return groupName;}
   const char * getParamName() {return paramName;}

private:
   char * groupName;
   char * paramName;
   ParameterSweepType type;
   int numValues;
   int currentBufferSize;
   double * valuesNumber;
   char ** valuesString;
};

class PVParams {
public:
   PVParams(size_t initialSize, InterColComm * icComm); // TODO Should be const InterColComm * comm
   PVParams(const char * filename, size_t initialSize, InterColComm * icComm);
   virtual ~PVParams();

   bool getParseStatus() { return parseStatus; }
   int   present(const char * groupName, const char * paramName);
   double value  (const char * groupName, const char * paramName);
   bool arrayPresent(const char * groupName, const char * paramName);
   double value  (const char * groupName, const char * paramName, double initialValue, bool warnIfAbsent=true);
   const float * arrayValues(const char * groupName, const char * paramName, int * arraySize, bool warnIfAbsent=true);
   const double * arrayValuesDbl(const char * groupName, const char * paramName, int * arraySize, bool warnIfAbsent=true);
   int   stringPresent(const char * groupName, const char * paramStringName);
   const char * stringValue(const char * groupName, const char * paramStringName, bool warnIfAbsent=true);
   ParameterGroup * group(const char * groupName);
   const char * groupNameFromIndex(int index);
   const char * groupKeywordFromIndex(int index);
   int warnUnread();
   bool hasBeenRead(const char * group_name, const char * param_name);
   bool presentAndNotBeenRead(const char * group_name, const char * param_name);
   int outputParams(FILE *);
   int setSweepValues(int n);

   void action_pvparams_directive(char * id, double val);
   void action_parameter_group(char * keyword, char * name);
   void action_parameter_def(char * id, double val);
   void action_parameter_array(char * id);
   void action_parameter_array_value(double val);
   void action_parameter_string_def(const char * id, const char * stringval);
   void action_parameter_filename_def(const char * id, const char * stringval);
   void action_include_directive(const char * stringval);
   void action_parameter_sweep(const char * id, const char * groupname, const char * paramname);
   void action_sweep_values_number(double val);
   void action_sweep_values_string(const char * stringval);
   void action_sweep_values_filename(const char * stringval);

   int numberOfGroups() {return numGroups;}
   InterColComm * getInterColComm() {return icComm;}
   int numberOfSweeps() {return numParamSweeps;}
   int getSweepSize() {return sweepSize;}

private:
   int parseStatus;
   int numGroups;
   size_t groupArraySize;
   // int maxGroups;
   ParameterGroup ** groups;
   ParameterStack * stack;
   ParameterArrayStack * arrayStack;
   ParameterStringStack * stringStack;
   bool debugParsing;
   bool disable;
   InterColComm * icComm;
   int getRank() {return icComm->commRank();}

   ParameterArray * currentParamArray;

   int numParamSweeps; // The number of different parameters that are changed during the sweep.
   ParameterSweep ** paramSweeps;
   ParameterSweep * activeParamSweep;
   int sweepSize; // The number of parameter value sets in the sweep.  Each ParameterSweep group in the params file must contain the same number of values, which is sweepSize.

   int initialize(size_t initialSize, InterColComm * icComm);
   int parsefile(const char * filename);
   int setSweepSize();
   void addGroup(char * keyword, char * name);
   void addActiveParamSweep(const char * group_name, const char * param_name);
   int checkDuplicates(const char * paramName);
   int newActiveParamSweep();
   int clearHasBeenReadFlags();
   static char * stripQuotationMarks(const char *s);
   static char * expandLeadingTilde(char *path);
};

}

#endif /* PVPARAMS_HPP_ */
