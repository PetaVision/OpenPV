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
#define PARAMETERSTRINGSTACK_INITIALCOUNT 5

// define for debug output
#define DEBUG_PARSING

/**
 * @yyin
 * @action_handler
 */
extern FILE* yyin;
int pv_parseParameters(PV::PVParams* action_handler);

#ifdef HAS_MAIN
#define INITIALNUMGROUPS 20   // maximum number of groups
int main()
{
   yyin = fopen("parser/params.txt", "r");
   PV::PVParams* handler = new PV::PVParams(INITIAL_NUM_GROUPS);

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
   hasBeenReadFlag = false;
}

Parameter::~Parameter()
{
   free(paramName);
}

/**
 * @name
 * @value
 */
ParameterString::ParameterString(const char * name, const char * value)
{
   paramName = (char *) malloc(strlen(name) + 1);
   strcpy(paramName,name);
   size_t valuelen = strlen(value)-2;  // strip quotes
   assert(value[0]=='"' && value[valuelen+1]=='"');
   paramValue = (char *) malloc(valuelen+1);
   strncpy(paramValue,value+1,valuelen);
   paramValue[valuelen] = '\0';
}

ParameterString::~ParameterString()
{
   free(paramName);
   free(paramValue);
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

/**
 * @name
 * @stack
 */
ParameterGroup::ParameterGroup(char * name, ParameterStack * stack, int rank)
{
   this->groupName = name;
   this->groupKeyword = NULL;
   this->stack     = stack;
   this->stringStack = NULL;
   this->processRank = rank;
}

ParameterGroup::~ParameterGroup()
{
   free(groupName);
   free(groupKeyword);
   delete stack;
   delete stringStack;
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
   int count = stack->size();
   for( int i=0; i<count; i++ ) {
      Parameter * p = stack->peek(i);
      if( !p->hasBeenRead() ) {
         if( processRank==0 ) fprintf(stderr,"Parameter group \"%s\": parameter \"%s\" has not been read.\n", name(), p->name());
         status = PV_FAILURE;
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
   fprintf(stderr, "Warning: FilenameDef is deprecated.  Use a string parameter inside a parameter group instead.\n"
                   "(getFilenameDef called with index=%d)\n",index);
   if( index >= count) return NULL;
   return filenamedefs[index];
}

FilenameDef * FilenameStack::getFilenameDefByKey(const char * searchKey) {
   fprintf(stderr, "Warning: FilenameDef is deprecated.  Use a string parameter inside a parameter group instead.\n"
                   "(getFilenameDefByKey called with searchKey=%s)\n",searchKey);
   for( unsigned int n = 0; n < count; n++) {
      if( !strcmp( searchKey, filenamedefs[n]->getKey() ) ) return filenamedefs[n];
   }
   return NULL;
}

int FilenameStack::push(FilenameDef * newfilenamedef) {
   if( count >= maxCount ) return PV_FAILURE;
   else {
      filenamedefs[count] = newfilenamedef;
      count++;
      return PV_SUCCESS;
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
 * @hc
 */
PVParams::PVParams(const char * filename, int initialSize, HyPerCol * hc)
{

   initialize(initialSize, hc);
   parsefile(filename);
}

/*
 * @initialSize
 * @hc
 */
PVParams::PVParams(int initialSize, HyPerCol * hc)
{
   initialize(initialSize, hc);
   parsefile(NULL);
}

PVParams::~PVParams()
{
   for( int i=0; i<numGroups; i++) {
      delete groups[i];
   }
   free(groups);
   delete stack;
   delete stringStack;
   delete fnstack;
}

/*
 * @initialSize
 */
int PVParams::initialize(int initialSize, HyPerCol * hc) {
   this->numGroups = 0;
   groupArraySize = initialSize;
   parentHyPerCol = hc;
#ifdef PV_USE_MPI
   InterColComm * icComm = hc->icCommunicator();
   rank = icComm->commRank();
#else
   rank = 0;
#endif // PV_USE_MPI

   groups = (ParameterGroup **) malloc(initialSize * sizeof(ParameterGroup *));
   stack = new ParameterStack(MAX_PARAMS);
   stringStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);
   fnstack = new FilenameStack(FILENAMESTACKMAXCOUNT);
#ifdef DEBUG_PARSING
   debugParsing = true;
#else
   debugParsing = false;
#endif//DEBUG_PARSING

   return ( groups && stack && stringStack && fnstack ) ? PV_SUCCESS : PV_FAILURE;
}

int PVParams::parsefile(const char * filename) {
   if (filename == NULL) {
      const char * altfile = INPUT_PATH "inparams.txt";
      printf("PVParams::PVParams: rank %d process opening alternate input file \"%s\"\n", rank, altfile);
      fflush(stdout);
      filename = altfile;
   }

   yyin = fopen(filename, "r");
   if (yyin == NULL) {
      fprintf(stderr, "PVParams::PVParams: rank %d process FAILED to open file %s.  Error code %d\n", rank, filename, errno);
      exit(errno);
   }

   parseStatus = pv_parseParameters(this);
   fclose(yyin);
   if( parseStatus != 0 ) {
      fprintf(stderr, "Rank %d process: pv_parseParameters failed with return value %d\n", rank, parseStatus);
   }
   return PV_SUCCESS;
}

/**
 * @groupName
 * @paramName
 */
int PVParams::present(const char * groupName, const char * paramName)
{
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      if( rank == 0 ) {
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
float PVParams::value(const char * groupName, const char * paramName)
{
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      if( rank == 0 ) {
         fprintf(stderr, "PVParams::value: ERROR, couldn't find a group for %s\n",
                 groupName);
      }
      exit(EXIT_FAILURE);
   }

   return g->value(paramName);
}

/**
 * @groupName
 * @paramName
 * @initialValue
 */
float PVParams::value(const char * groupName, const char * paramName, float initialValue, bool warnIfAbsent)
{
   if (present(groupName, paramName)) {
      return value(groupName, paramName);
   }
   else {
      if( warnIfAbsent && rank == 0 ) {
          printf("Using default value %f for parameter \"%s\" in group \"%s\"\n",initialValue, paramName, groupName);
      }
      return initialValue;
   }
}

/*
 *  @groupName
 *  @paramStringName
 */
int PVParams::stringPresent(const char * groupName, const char * paramStringName) {
   ParameterGroup * g = group(groupName);
   if (g == NULL) {
      if( rank == 0 ) {
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
      if( warnIfAbsent && rank == 0 ) {
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

   // Verify that the new group's name is not an existing group's name
   for( int k=0; k<numGroups; k++ ) {
      if( !strcmp(name, groups[k]->name())) {
         fprintf(stderr, "Rank %d process: group name \"%s\" duplicated\n", rank, name);
         exit(EXIT_FAILURE);
      }
   }

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

   groups[numGroups] = new ParameterGroup(name, stack, rank);
   groups[numGroups]->setGroupKeyword(keyword);
   groups[numGroups]->setStringStack(stringStack);

   // the parameter group takes over control of the PVParams's stack and stringStack; make new ones.
   stack = new ParameterStack(MAX_PARAMS);
   stringStack = new ParameterStringStack(PARAMETERSTRINGSTACK_INITIALCOUNT);

   numGroups++;
}

int PVParams::warnUnread() {
   int status = PV_SUCCESS;
   for( int i=0; i<numberOfGroups(); i++) {
      if( groups[i]->warnUnread() == PV_FAILURE) {
         status = PV_FAILURE;
      }
   }
   return status;
}

/**
 * @val
 */
void PVParams::action_pvparams_directive(char * id, double val)
{
   if( !strcmp(id,"debugParsing") ) {
      debugParsing = (val != 0);
      if( rank == 0 ) {
         printf("debugParsing turned ");
         if(debugParsing) {
            printf("on.\n");
         }
         else {
            printf("off.\n");
         }
      }
   }
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

   if(debugParsing && rank==0 ) {
      printf("action_parameter_group: %s \"%s\" parsed successfully.\n", keyword, name);
      fflush(stdout);
   }

   // build a parameter group
   addGroup(keyword, name);
}

/**
 * @id
 * @val
 */
void PVParams::action_parameter_def(char * id, double val)
{
   if(debugParsing && rank == 0) {
      fflush(stdout);
      printf("action_parameter_def: %s = %lf\n", id, val);
      fflush(stdout);
   }
   if( checkDuplicates(id) != PV_SUCCESS ) exit(EXIT_FAILURE);
   Parameter * p = new Parameter(id, val);
   stack->push(p);
}

void PVParams::action_parameter_string_def(const char * id, const char * stringval) {
   if( debugParsing && rank == 0 ) {
      fflush(stdout);
      printf("action_parameter_string_def: %s = %s\n", id, stringval);
      fflush(stdout);
   }
   if( checkDuplicates(id) != PV_SUCCESS ) exit(EXIT_FAILURE);
   ParameterString * pstr = new ParameterString(id, stringval);
   stringStack->push(pstr);
}

int PVParams::checkDuplicates(const char * paramName) {
   int status = PV_SUCCESS;
   for( int k=0; k<stack->size(); k++ ) {
      if( !strcmp(paramName, stack->peek(k)->name() ) ) {
         fprintf(stderr, "Rank %d process: parameter name \"%s\" duplicates a previous parameter name\n", rank, paramName);
         status = PV_FAILURE;
         break;
      }
   }
   for( int k=0; k<stringStack->size(); k++ ) {
      if( !strcmp(paramName, stringStack->peek(k)->getName() ) ) {
         fprintf(stderr, "Rank %d process: parameter name \"%s\" duplicates a previous string parameter name\n", rank, paramName);
         status = PV_FAILURE;
         break;
      }
   }
   if( status != PV_SUCCESS ) {
      if( numberOfGroups() == 0 ) {
         fprintf(stderr, "Rank %d process: this is the first parameter group being parsed\n", rank);
      }
      else {
         fprintf(stderr, "Rank %d process: last parameter group successfully added was \"%s\"\n", rank, groups[numberOfGroups()-1]->name());
      }
   }
   return status;
}

// action_filename_def deprecated on Oct 27, 2011
void PVParams::action_filename_def(char * id, char * path)
{
   if( rank == 0 ) {
      fprintf(stderr, "Warning: FilenameDef is deprecated.  Use a string parameter inside a parameter group instead.\n"
                      "(action_filename_def called with id=%s, path=%s)\n",id,path);
   }
   if( debugParsing && rank == 0 ) {
      fflush(stdout);
      printf("action_filename_def: %s = %s\n", id, path);
      fflush(stdout);
   }
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
   if( status != PV_SUCCESS) {
      fprintf(stderr, "Rank %d process: No room for %s:%s\n", rank, label, path);
   }
}  // close PVParams::action_filename_def block

}  // close namespace PV block
