/*
 * PVParams.hpp
 *
 *  Created on: Nov 27, 2008
 *      Author: rasmussn
 */

#ifndef PVPARAMS_HPP_
#define PVPARAMS_HPP_

#include "../include/pv_common.h"

// TODO - make MAX_PARAMS dynamic
#define MAX_PARAMS 100  // maximum number of parameters in a group

#undef HAS_MAIN   // define if provides a main function

namespace PV {

class Parameter {
public:
   Parameter(char * name, double value);
   virtual ~Parameter();

   const char * name()      { return paramName; }
   double value()           { return paramValue; }

private:
   char * paramName;
   double paramValue;
};

class ParameterString {
public:
   ParameterString(const char * name, const char * value);
   virtual ~ParameterString();

   const char * getName()      { return paramName; }
   const char * getValue()           { return paramValue; }

private:
   char * paramName;
   char * paramValue;
};

class ParameterStack {
public:
   ParameterStack(int maxCount);
   virtual ~ParameterStack();

   int push(Parameter * param);
   Parameter * pop();
   Parameter * peek(int index)   { return parameters[index]; }
   int size()                    { return count; }

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

private:
   int count;
   int allocation;
   ParameterString ** parameterStrings;
};

class ParameterGroup {
public:
   ParameterGroup(char * name, ParameterStack * stack);
   virtual ~ParameterGroup();

   const char * name()   { return groupName; }
   const char * getGroupKeyword() { return groupKeyword; }
   int setGroupKeyword(const char * keyword);
   int setStringStack(ParameterStringStack * stringStack);
   int   present(const char * name);
   float value  (const char * name);
   int   stringPresent(const char * stringName);
   const char * stringValue(const char * stringName);

private:
   char * groupName;
   char * groupKeyword;
   ParameterStack * stack;
   ParameterStringStack * stringStack;
};

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
   PVParams(int initialSize);
   PVParams(const char * filename, int initialSize);
   virtual ~PVParams();

   int   present(const char * groupName, const char * paramName);
   float value  (const char * groupName, const char * paramName);
   float value  (const char * groupName, const char * paramName, float initialValue);
   int   stringPresent(const char * groupName, const char * paramStringName);
   const char * stringValue(const char * groupName, const char * paramStringName);
   ParameterGroup * group(const char * groupName);
   const char * groupNameFromIndex(int index);
   const char * groupKeywordFromIndex(int index);
   const char * getFilename(const char * id);

   void action_parameter_group(char * keyword, char * name);
   void action_parameter_def(char * id, double val);
   void action_parameter_string_def(const char * id, const char * stringval);
   void action_filename_def(char * id, char * path); // Deprecate?
   int numberOfGroups() {return numGroups;}

private:
   int numGroups;
   int groupArraySize;
   // int maxGroups;
   ParameterGroup ** groups;
   ParameterStack * stack;
   ParameterStringStack * stringStack;
   FilenameStack * fnstack; // Deprecate?

   int initialize(int initialSize);
   void addGroup(char * keyword, char * name);
};

}

#endif /* PVPARAMS_HPP_ */
