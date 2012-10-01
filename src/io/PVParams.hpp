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
#include <string.h>

// TODO - make MAX_PARAMS dynamic
#define MAX_PARAMS 100  // maximum number of parameters in a group

#undef HAS_MAIN   // define if provides a main function

namespace PV {

class Parameter {
public:
   Parameter(const char * name, double value);
   virtual ~Parameter();

   const char * name()      { return paramName; }
   double value()           { hasBeenReadFlag = true; return paramValue; }
   bool hasBeenRead()       { return hasBeenReadFlag; }
   int outputParam(FILE * fp, int indentation);
   void clearHasBeenRead()    { hasBeenReadFlag = false; }
   void setValue(double v)  { paramValue = v; }

private:
   char * paramName;
   double paramValue;
   bool   hasBeenReadFlag;
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
#ifdef OBSOLETE // Marked obsolete Aug 10, 2012.  String parameters should be handled on an equal footing with numerical parameters
   ParameterGroup(char * name, ParameterStack * stack, int rank=0);
#endif // OBSOLETE
   ParameterGroup(char * name, ParameterStack * stack, ParameterStringStack * string_stack, int rank=0);
   virtual ~ParameterGroup();

   const char * name()   { return groupName; }
   const char * getGroupKeyword() { return groupKeyword; }
   int setGroupKeyword(const char * keyword);
   int setStringStack(ParameterStringStack * stringStack);
   int   present(const char * name);
   float value  (const char * name);
   int   stringPresent(const char * stringName);
   const char * stringValue(const char * stringName);
   int warnUnread();
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
   int pushNumericValue(float val);
   int pushStringValue(const char * sval);
   int getNumValues() {return numValues;}
   ParameterSweepType getType() {return type;}
   int getNumericValue(int n, float * val);
   const char * getStringValue(int n);
   const char * getGroupName() {return groupName;}
   const char * getParamName() {return paramName;}

private:
   char * groupName;
   char * paramName;
   ParameterSweepType type;
   int numValues;
   int currentBufferSize;
   float * valuesNumber;
   char ** valuesString;
};

#ifdef OBSOLETE // Marked obsolete Aug 9, 2012.  No one uses this, and filenames can be defined as string parameters in parameter groups
// FilenameDef and FilenameStack deprecated Oct 27, 2011
class FilenameDef {
public:
   FilenameDef(char * newKey, char * newValue);
   virtual ~FilenameDef();

   const char * getKey() { return key; };
   const char * getValue() { return value; };

private:
   char * key;
   char * value;
};

// FilenameDef and FilenameStack deprecated Oct 27, 2011
class FilenameStack {
public:
   FilenameStack(unsigned int maxCount);
   virtual ~FilenameStack();

   unsigned int getMaxCount() { return maxCount; };
   unsigned int getCount() { return count; };
   FilenameDef * getFilenameDef(unsigned int index);
   FilenameDef * getFilenameDefByKey(const char * searchKey);
   int push(FilenameDef * newfilenamedef);
   FilenameDef * pop();
private:
   unsigned int maxCount;
   unsigned int count;
   FilenameDef ** filenamedefs;
};
#endif // OBSOLETE

class PVParams {
public:
   PVParams(int initialSize, InterColComm * icComm); // TODO Should be const InterColComm * comm
   PVParams(const char * filename, int initialSize, InterColComm * icComm);
   virtual ~PVParams();

   bool getParseStatus() { return parseStatus; }
   int   present(const char * groupName, const char * paramName);
   float value  (const char * groupName, const char * paramName);
   float value  (const char * groupName, const char * paramName, float initialValue, bool warnIfAbsent=true);
   int   stringPresent(const char * groupName, const char * paramStringName);
   const char * stringValue(const char * groupName, const char * paramStringName, bool warnIfAbsent=true);
   ParameterGroup * group(const char * groupName);
   const char * groupNameFromIndex(int index);
   const char * groupKeywordFromIndex(int index);
#ifdef OBSOLETE // Marked obsolete Aug 9, 2012.  No one uses this, and filenames can be defined as string parameters in parameter groups
   const char * getFilename(const char * id);
#endif // OBSOLETE
   int warnUnread();
   int outputParams(FILE *);
   int setSweepValues(int n);

   void action_pvparams_directive(char * id, double val);
   void action_parameter_group(char * keyword, char * name);
   void action_parameter_def(char * id, double val);
   void action_parameter_string_def(const char * id, const char * stringval);
   void action_parameter_filename_def(const char * id, const char * stringval);
   void action_include_directive(const char * stringval);
#ifdef OBSOLETE // Marked obsolete March 15, 2012.  There's more flexibility in defining string parameters within groups
   void action_filename_def(char * id, char * path); // Deprecated Oct 27, 2011
#endif // OBSOLETE
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
   int groupArraySize;
   // int maxGroups;
   ParameterGroup ** groups;
   ParameterStack * stack;
   ParameterStringStack * stringStack;
#ifdef OBSOLETE // Marked obsolete Aug 9, 2012.  No one uses this, and filenames can be defined as string parameters in parameter groups
   FilenameStack * fnstack; // Deprecated Oct 27, 2011
#endif // OBSOLETE
   bool debugParsing;
   InterColComm * icComm;
   int getRank() {return icComm->commRank();}

   int numParamSweeps; // The number of different parameters that are changed during the sweep.
   ParameterSweep ** paramSweeps;
   ParameterSweep * activeParamSweep;
   int sweepSize; // The number of parameter value sets in the sweep.  Each ParameterSweep group in the params file must contain the same number of values, which is sweepSize.

   int initialize(int initialSize, InterColComm * icComm);
   int parsefile(const char * filename);
   int setSweepSize();
   void addGroup(char * keyword, char * name);
   void addActiveParamSweep(const char * group_name, const char * param_name);
   int checkDuplicates(const char * paramName);
   int newActiveParamSweep();
   int clearHasBeenReadFlags();
   static char * stripQuotationMarks(const char *s);
};

}

#endif /* PVPARAMS_HPP_ */
