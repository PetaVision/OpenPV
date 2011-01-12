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

#define FILENAMESTACKMAXCOUNT 10

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

FilenameDef::FilenameDef(char * newKey, char * newValue) {
   size_t keylen, valuelen;

   keylen = strlen(newKey);
   key = (char *) malloc(sizeof(char)*(keylen+1));
   if( !key ) return;
   strcpy(key, newKey);
   valuelen = strlen(newValue);
   value = (char *) malloc(sizeof(char)*(valuelen+1));
   if( !value )
   {
      free(key);
      return;
   }
   strcpy(value, newValue);
}

FilenameDef::~FilenameDef() {
   free(key);
   free(value);
}

FilenameStack::FilenameStack(unsigned int maxCount) {
   this->maxCount = maxCount;
   this->count = 0;
   this->filenamedefs = (FilenameDef **) calloc(maxCount, sizeof(FilenameDef *) );
   if( !(this->filenamedefs) ) this->maxCount = 0;
}

FilenameStack::~FilenameStack() {
   for( unsigned int n = 0; n < maxCount; n++) {
      if( filenamedefs[n] ) delete filenamedefs[n];
   }
   if( this->filenamedefs ) free( this->filenamedefs );
}

FilenameDef * FilenameStack::getFilenameDef(unsigned int index) {
   if( index >= count) return NULL;
   return filenamedefs[index];
}

FilenameDef * FilenameStack::getFilenameDefByKey(const char * searchKey) {
   for( unsigned int n = 0; n < maxCount; n++) {
      if( !strcmp( searchKey, filenamedefs[n]->getKey() ) ) return filenamedefs[n];
   }
   return NULL;
}

int FilenameStack::push(FilenameDef * newfilenamedef) {
   if( count >= maxCount ) return EXIT_FAILURE;
   else {
      filenamedefs[count] = newfilenamedef;
      count++;
      return EXIT_SUCCESS;
   }
}

FilenameDef * FilenameStack::pop() {
   if( count == 0) return NULL;
   count--;
   return filenamedefs[count];
}


/**
 * @filename
 * @initialSize
 */
PVParams::PVParams(const char * filename, int initialSize)
{
   const char * altfile = INPUT_PATH "inparams.txt";

   this->numGroups = 0;
   groupArraySize = initialSize;

   groups = (ParameterGroup **) malloc(initialSize * sizeof(ParameterGroup *));
   stack = new ParameterStack(MAX_PARAMS);
   fnstack = new FilenameStack(FILENAMESTACKMAXCOUNT);

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
 * @initialSize
 */
PVParams::PVParams(int initialSize)
{
   this->numGroups = 0;
   groupArraySize = initialSize;

   groups = (ParameterGroup **) malloc(initialSize * sizeof(ParameterGroup *));
   stack = new ParameterStack(MAX_PARAMS);
   fnstack = new FilenameStack(FILENAMESTACKMAXCOUNT);
}

PVParams::~PVParams()
{
   free(groups);
   delete stack;
   delete fnstack;
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

const char * PVParams::getFilename(const char * id)
{
   FilenameDef * fd = fnstack->getFilenameDefByKey(id);
   const char * fn = (fd != NULL) ? fd->getValue() : NULL;

   return fn;
}

/**
 * @keyword
 * @name
 */
void PVParams::addGroup(char * keyword, char * name)
{
   assert(numGroups <= groupArraySize);
   if( numGroups == groupArraySize ) {
      groupArraySize += RESIZE_ARRAY_INCR;
      ParameterGroup ** newGroups = (ParameterGroup **) malloc( groupArraySize * sizeof(ParameterGroup *) );
      assert(newGroups);
      for(  int k=0; k< numGroups; k++ ) {
         newGroups[k] = groups[k];
      }
      free(groups);
      groups = newGroups;
   }

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

void PVParams::action_filename_def(char * id, char * path)
{
#ifdef DEBUG_PARSING
   fflush(stdout);
   printf("action_filename_decl: %s = %s\n", id, path);
   fflush(stdout);
#endif
   size_t pathlength = strlen( path );
   assert( pathlength >= 2 ); // path still includes the delimiting quotation marks
   char * filenameptr = (char *) malloc( sizeof(char)*( pathlength - 1) );
   assert( filenameptr != NULL );
   strncpy( filenameptr, &(path[1]), pathlength - 2);
   filenameptr[pathlength-2] = 0;

   size_t labellength = strlen( id );
   assert( labellength >= 2 );
   char * label = (char *) malloc( sizeof(char)*( pathlength - 1) );
   assert( label != NULL );
   strncpy( label, &(id[1]), labellength - 2);
   label[labellength-2] = 0;

   FilenameDef * fndef = new FilenameDef(label, filenameptr);
   int status = fnstack->push(fndef);
   if( status != EXIT_SUCCESS) fprintf(stderr, "No room for %s:%s\n", label, path);
}  // close PVParams::action_filename_def block

}  // close namespace PV block
