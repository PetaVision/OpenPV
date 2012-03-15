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

// TODO - make MAX_PARAMS dynamic
#define MAX_PARAMS 100  // maximum number of parameters in a group

#undef HAS_MAIN   // define if provides a main function

namespace PV {

class Parameter {
public:
   Parameter(char * name, double value);
   virtual ~Parameter();

   const char * name()      { return paramName; }
   double value()           { hasBeenReadFlag = true; return paramValue; }
   bool hasBeenRead()       { return hasBeenReadFlag; }
   int outputParam(FILE * fp, int indentation);

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
   ParameterGroup(char * name, ParameterStack * stack, int rank=0);
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
   int outputGroup(FILE * fp);

private:
   char * groupName;
   char * groupKeyword;
   ParameterStack * stack;
   ParameterStringStack * stringStack;
   int processRank;
};

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

class PVParams {
public:
   PVParams(int initialSize, HyPerCol * hc); // TODO Should be const HyPerCol, but we need hc->icComm() and icComm() isn't constant
   PVParams(const char * filename, int initialSize, HyPerCol * hc);
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
   const char * getFilename(const char * id);
   int warnUnread();
   int outputParams(FILE *);

   void action_pvparams_directive(char * id, double val);
   void action_parameter_group(char * keyword, char * name);
   void action_parameter_def(char * id, double val);
   void action_parameter_string_def(const char * id, const char * stringval);
   void action_include_directive(const char * stringval);
#ifdef OBSOLETE // Marked obsolete March 15, 2012.  There's more flexibility in defining string parameters within groups
   void action_filename_def(char * id, char * path); // Deprecated Oct 27, 2011
#endif // OBSOLETE
   int numberOfGroups() {return numGroups;}

private:
   int parseStatus;
   int numGroups;
   int groupArraySize;
   // int maxGroups;
   ParameterGroup ** groups;
   ParameterStack * stack;
   ParameterStringStack * stringStack;
   FilenameStack * fnstack; // Deprecated Oct 27, 2011
   bool debugParsing;
   HyPerCol * parentHyPerCol; // TODO Should be const; see comment on prototype for constructor
   int rank;

   int initialize(int initialSize, HyPerCol * hc);
   int parsefile(const char * filename);
   void addGroup(char * keyword, char * name);
   int checkDuplicates(const char * paramName);
};

}

#endif /* PVPARAMS_HPP_ */
