/*
 * PVParams.hpp
 *
 *  Created on: Nov 27, 2008
 *      Author: rasmussn
 */

#ifndef PVPARAMS_HPP_
#define PVPARAMS_HPP_

// TODO - make number dynamic
#define MAX_GROUPS 20   // maximum number of groups
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

class ParameterGroup {
public:
   ParameterGroup(char * name, ParameterStack * stack);
   virtual ~ParameterGroup();

   const char * name()   { return groupName; }

   int   present(const char * name);
   float value  (const char * name);

private:
   char * groupName;
   ParameterStack * stack;
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
   PVParams(int maxGroups);
   PVParams(const char * filename, int maxGroups);
   virtual ~PVParams();

   int   present(const char * groupName, const char * paramName);
   float value  (const char * groupName, const char * paramName);
   float value  (const char * groupName, const char * paramName, float initialValue);
   ParameterGroup * group(const char * groupName);
   const char * getFilename(const char * id);

   void action_parameter_group(char * keyword, char * name);
   void action_parameter_def(char * id, double val);
   void action_filename_def(char * id, char * path);

private:
   int numGroups;
   int maxGroups;
   ParameterGroup ** groups;
   ParameterStack * stack;
   FilenameStack * fnstack;

   void addGroup(char * keyword, char * name);
};

}

#endif /* PVPARAMS_HPP_ */
