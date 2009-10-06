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

// define for debug output
#undef DEBUG_PARSING

/**
 * @yyin
 * @action_handler
 */
extern FILE* yyin;
int pv_parseParameters(PV::PVParams* action_handler);

#ifdef HAS_MAIN
int main()
{
   yyin = fopen("parser/params.txt", "r");
   PV::PVParams* handler = new PV::PVParams(MAX_GROUPS);

   pv_parseParameters(handler);

   fclose(yyin);
   delete handler;

   return 0;
}
#endif // HAS_MAIN

namespace PV {

/**
 * @name
 * @value
 */
Parameter::Parameter(char * name, double value)
{
   paramName  = name;
   paramValue = value;
}

Parameter::~Parameter()
{
   free(paramName);
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

/**
 * @name
 * @stack
 */
ParameterGroup::ParameterGroup(char * name, ParameterStack * stack)
{
   this->groupName = name;
   this->stack     = stack;
}

ParameterGroup::~ParameterGroup()
{
   free(groupName);
   delete stack;
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
         return 1;
      }
   }
   return 0;
}

/**
 * @name
 */
float ParameterGroup::value(const char * name)
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

/**
 * @filename
 * @maxGroups
 */
PVParams::PVParams(const char * filename, int maxGroups)
{
   const char * altfile = INPUT_PATH "inparams.txt";

   this->numGroups = 0;
   this->maxGroups = maxGroups;

   groups = (ParameterGroup **) malloc(maxGroups * sizeof(ParameterGroup *));
   stack = new ParameterStack(MAX_PARAMS);

   if (filename == NULL) {
      printf("PVParams::PVParams: trying to open alternate input file %s\n", altfile);
      fflush(stdout);
      filename = altfile;
   }

   yyin = fopen(filename, "r");
   if (yyin == NULL) {
      fprintf(stderr, "PVParams::PVParams: FAILED to open file %s\n", filename);
      exit(1);
   }
   pv_parseParameters(this);
   fclose(yyin);
}

/**
 * @maxGroups
 */
PVParams::PVParams(int maxGroups)
{
   this->numGroups = 0;
   this->maxGroups = maxGroups;

   groups = (ParameterGroup **) malloc(maxGroups * sizeof(ParameterGroup *));
   stack = new ParameterStack(MAX_PARAMS);
}

PVParams::~PVParams()
{
   free(groups);
   delete stack;
}

/**
 * @groupName
 * @paramName
 */
int PVParams::present(const char * groupName, const char * paramName)
{
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      fprintf(stderr, "PVParams::value: ERROR, couldn't find a group for %s\n",
              groupName);
      exit(1);
   }

   return g->present(paramName);
}

/**
 * @groupName
 * @paramName
 */
float PVParams::value(const char * groupName, const char * paramName)
{
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      fprintf(stderr, "PVParams::value: ERROR, couldn't find a group for %s\n",
              groupName);
      exit(1);
   }

   return g->value(paramName);
}

/**
 * @groupName
 * @paramName
 * @initialValue
 */
float PVParams::value(const char * groupName, const char * paramName, float initialValue)
{
   if (present(groupName, paramName)) {
      return value(groupName, paramName);
   }
   else {
      return initialValue;
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

/**
 * @keyword
 * @name
 */
void PVParams::addGroup(char * keyword, char * name)
{
   assert(numGroups < maxGroups);
   groups[numGroups++] = new ParameterGroup(name, stack);

   // the parameter group takes over control of the stack
   stack = new ParameterStack(MAX_PARAMS);

   free(keyword);   // not used
}

/**
 * @keyword
 * @name
 */
void PVParams::action_parameter_group(char * keyword, char * name)
{
   // remove surrounding quotes
   int len = strlen(++name);
   name[len-1] = '\0';

#ifdef DEBUG_PARSING
   fflush(stdout);
   printf("action_parameter_group: %s \"%s\" = \n", keyword, name);
   fflush(stdout);
#endif

   // build a parameter group
   addGroup(keyword, name);
}

/**
 * @id
 * @val
 */
void PVParams::action_parameter_def(char * id, double val)
{
#ifdef DEBUG_PARSING
   fflush(stdout);
   printf("action_parameter_def: %s = %lf\n", id, val);
   fflush(stdout);
#endif
   Parameter * p = new Parameter(id, val);
   stack->push(p);
}

}
