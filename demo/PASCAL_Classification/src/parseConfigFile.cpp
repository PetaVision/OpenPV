/*
 * parseConfigFile.cpp
 *
 *  Created on: Sep 24, 2015
 *      Author: pschultz
 */

#include "parseConfigFile.hpp"
#include <include/cMakeHeader.h>
#include "cMakeHeader.h"

int parseConfigFile(PV::InterColComm * icComm, char ** imageLayerNamePtr, char ** resultLayerNamePtr, char ** resultTextFilePtr, char ** octaveCommandPtr, char ** octaveLogFilePtr, char ** classNamesPtr, char ** evalCategoryIndicesPtr, char ** displayCategoryIndicesPtr, char ** highlightThresholdPtr, char ** heatMapThresholdPtr, char ** heatMapMaximumPtr, char ** drawBoundingBoxesPtr, char ** boundingBoxThicknessPtr, char ** dbscanEpsPtr, char ** dbscanDensityPtr, char ** heatMapMontageDirPtr, char ** displayCommandPtr);
int parseConfigParameter(PV::InterColComm * icComm, char const * inputLine, char const * configParameter, char ** parameterPtr, unsigned int lineNumber);
int checkOctaveArgumentString(char const * argString, char const * argName);
int checkOctaveArgumentNumeric(char const * argString, char const * argName);
int checkOctaveArgumentVector(char const * argString, char const * argName);


int parseConfigFile(PV::InterColComm * icComm, char ** imageLayerNamePtr, char ** resultLayerNamePtr, char ** resultTextFilePtr, char ** octaveCommandPtr, char ** octaveLogFilePtr, char ** classNamesPtr, char ** evalCategoryIndicesPtr, char ** displayCategoryIndicesPtr, char ** highlightThresholdPtr, char ** heatMapThresholdPtr, char ** heatMapMaximumPtr, char ** drawBoundingBoxesPtr, char ** boundingBoxThicknessPtr, char ** dbscanEpsPtr, char ** dbscanDensityPtr, char ** heatMapMontageDirPtr, char ** displayCommandPtr)
{
   // Under MPI, all processes must call this function in parallel, but only the root process does I/O
   int status = PV_SUCCESS;
   FILE * parseConfigFileFP = NULL;
   if (icComm->commRank()==0) {
      parseConfigFileFP = fopen(CONFIG_FILE, "r");
      if (parseConfigFileFP == NULL)
      {
         fprintf(stderr, "Unable to open config file \"%s\": %s\n", CONFIG_FILE, strerror(errno));
         return PV_FAILURE;
      }
   }
   *imageLayerNamePtr = NULL;
   *resultLayerNamePtr = NULL;
   *resultTextFilePtr = NULL;
   *octaveCommandPtr = NULL;
   *octaveLogFilePtr = NULL;
   *classNamesPtr = NULL;
   *evalCategoryIndicesPtr = NULL;
   *displayCategoryIndicesPtr = NULL;
   *highlightThresholdPtr = NULL;
   *heatMapThresholdPtr = NULL;
   *heatMapMaximumPtr = NULL;
   *drawBoundingBoxesPtr = NULL;
   *boundingBoxThicknessPtr = NULL;
   *dbscanEpsPtr = NULL;
   *dbscanDensityPtr = NULL;
   *heatMapMontageDirPtr = NULL;
   *displayCommandPtr = NULL;
   struct fgetsresult { char contents[TEXTFILEBUFFERSIZE]; char * result; };
   struct fgetsresult line;
   unsigned int linenumber=0;
   while (true)
   {
      linenumber++;
      if (icComm->commRank()==0) {
         line.result = fgets(line.contents, TEXTFILEBUFFERSIZE, parseConfigFileFP);
      }
      MPI_Bcast(&line, sizeof(line), MPI_CHAR, 0, icComm->communicator());
      if (icComm->commRank()!=0 && line.result!=NULL) {
         line.result = line.contents;
      }
      if (line.result==NULL) { break; }
      char * colonsep = strchr(line.result,':');
      if (colonsep==NULL) { break; }
      char * openquote = strchr(colonsep,'"');
      if (openquote==NULL) { break; }
      char * closequote = strchr(openquote+1,'"');
      if (closequote==NULL) { break; }
      *colonsep='\0';
      *openquote='\0';
      *closequote='\0';
      char * keyword = line.contents;
      char * value = &openquote[1];

      if (!strcmp(keyword,"imageLayer"))
      {
         status = parseConfigParameter(icComm, keyword, value, imageLayerNamePtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"resultLayer"))
      {
         status = parseConfigParameter(icComm, keyword, value, resultLayerNamePtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"resultTextFile"))
      {
         status = parseConfigParameter(icComm, keyword, value, resultTextFilePtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"octaveCommand"))
      {
         status = parseConfigParameter(icComm, keyword, value, octaveCommandPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"octaveLogFile"))
      {
         status = parseConfigParameter(icComm, keyword, value, octaveLogFilePtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"classNames"))
      {
         status = parseConfigParameter(icComm, keyword, value, classNamesPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"evalCategoryIndices"))
      {
         status = parseConfigParameter(icComm, keyword, value, evalCategoryIndicesPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"displayCategoryIndices"))
      {
         status = parseConfigParameter(icComm, keyword, value, displayCategoryIndicesPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"highlightThreshold"))
      {
         status = parseConfigParameter(icComm, keyword, value, highlightThresholdPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"heatMapThreshold"))
      {
         status = parseConfigParameter(icComm, keyword, value, heatMapThresholdPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"heatMapMaximum"))
      {
         status = parseConfigParameter(icComm, keyword, value, heatMapMaximumPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"drawBoundingBoxes"))
      {
         status = parseConfigParameter(icComm, keyword, value, drawBoundingBoxesPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"boundingBoxThickness"))
      {
         status = parseConfigParameter(icComm, keyword, value, boundingBoxThicknessPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"dbscanEps"))
      {
         status = parseConfigParameter(icComm, keyword, value, dbscanEpsPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"dbscanDensity"))
      {
         status = parseConfigParameter(icComm, keyword, value, dbscanDensityPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"heatMapMontageDir"))
      {
         status = parseConfigParameter(icComm, keyword, value, heatMapMontageDirPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
      if (!strcmp(keyword,"displayCommand"))
      {
         status = parseConfigParameter(icComm, keyword, value, displayCommandPtr, linenumber);
         if (status != PV_SUCCESS) { break; }
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*imageLayerNamePtr==NULL)
      {
         fprintf(stderr, "imageLayer was not defined in %s.\n", CONFIG_FILE);
         status = PV_FAILURE;
      }
      else
      {
         status = checkOctaveArgumentString(*imageLayerNamePtr, "imageLayer");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*resultLayerNamePtr==NULL)
      {
         fprintf(stderr, "resultLayer was not defined in %s.\n", CONFIG_FILE);
         status = PV_FAILURE;
      }
      else
      {
         status = checkOctaveArgumentString(*resultLayerNamePtr, "resultLayer");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*resultTextFilePtr==NULL)
      {
         if (icComm->commRank()==0) {
            fprintf(stderr, "resultTextFile was not defined in %s; a text file of results will not be produced.\n", CONFIG_FILE);
         }
         *resultTextFilePtr = strdup("");
      }
      else
      {
         status = checkOctaveArgumentString(*resultTextFilePtr, "resultTextFile");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*octaveCommandPtr==NULL)
      {
         fprintf(stderr, "octaveCommand was not defined in %s.\n", CONFIG_FILE);
         status = PV_FAILURE;
      }
      else
      {
         status = checkOctaveArgumentString(*octaveCommandPtr, "octaveCommand");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*octaveLogFilePtr==NULL)
      {
         fprintf(stderr, "octaveLogFile was not defined in %s.\n", CONFIG_FILE);
         status = PV_FAILURE;
      }
      else
      {
         status = checkOctaveArgumentString(*octaveLogFilePtr, "octaveLogFile");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*classNamesPtr==NULL)
      {
         if (icComm->commRank()==0) {
            fprintf(stderr, "classNames was not defined in %s; setting class names to feature indices.\n", CONFIG_FILE);
         }
         *classNamesPtr = strdup("{}");
      }
      else
      {
         status = checkOctaveArgumentString(*classNamesPtr, "classNames");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*evalCategoryIndicesPtr==NULL)
      {
         if (icComm->commRank()==0) {
            printf("evalCategoryIndices was not defined in %s; using all indices.\n", CONFIG_FILE);
         }
         *evalCategoryIndicesPtr = strdup("[]");
      }
      else
      {
         status = checkOctaveArgumentVector(*evalCategoryIndicesPtr, "evalCategoryIndices");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*displayCategoryIndicesPtr==NULL)
      {
         if (icComm->commRank()==0) {
            printf("evalCategoryIndices was not defined in %s; using all indices.\n", CONFIG_FILE);
         }
         *displayCategoryIndicesPtr = strdup("[]");
      }
      else
      {
         status = checkOctaveArgumentVector(*displayCategoryIndicesPtr, "displayCategoryIndices");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*highlightThresholdPtr==NULL)
      {
         if (icComm->commRank()==0) {
            printf("highlightThreshold was not defined in %s; setting to zero\n", CONFIG_FILE);
         }
         *highlightThresholdPtr = strdup("0.0");
      }
      else
      {
         status = checkOctaveArgumentNumeric(*highlightThresholdPtr, "highlightThreshold");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*heatMapThresholdPtr==NULL)
      {
         if (icComm->commRank()==0) {
            printf("heatMapThreshold was not defined in %s; setting to same as highlightThreshold\n", CONFIG_FILE);
         }
         *heatMapThresholdPtr = strdup(*highlightThresholdPtr);
      }
      else
      {
         status = checkOctaveArgumentNumeric(*heatMapThresholdPtr, "heatMapThreshold");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*heatMapMaximumPtr==NULL)
      {
         if (icComm->commRank()==0) {
            printf("heatMapMaximum was not defined in %s; setting to 1.0\n", CONFIG_FILE);
         }
         *heatMapMaximumPtr = strdup("1.0");
      }
      else
      {
         status = checkOctaveArgumentNumeric(*heatMapMaximumPtr, "heatMapMaximum");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*drawBoundingBoxesPtr==NULL)
      {
         if (icComm->commRank()==0) {
            printf("drawBoundingBoxes was not defined in %s; setting to 0 (False)\n", CONFIG_FILE);
         }
         *drawBoundingBoxesPtr = strdup("0.0");
      }
      else
      {
         status = checkOctaveArgumentNumeric(*drawBoundingBoxesPtr, "drawBoundingBoxes");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*boundingBoxThicknessPtr==NULL)
      {
         if (icComm->commRank()==0) {
            printf("boundingBoxThickness was not defined in %s; setting to 5\n", CONFIG_FILE);
         }
         *boundingBoxThicknessPtr = strdup("5.0");
      }
      else
      {
         status = checkOctaveArgumentNumeric(*boundingBoxThicknessPtr, "boundingBoxThickness");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*dbscanEpsPtr==NULL)
      {
         if (icComm->commRank()==0) {
            printf("dbscanEps was not defined in %s; dbscan will attempt to calculate it\n", CONFIG_FILE);
         }
         *dbscanEpsPtr = strdup("[]");
      }
      else
      {
         status = checkOctaveArgumentNumeric(*dbscanEpsPtr, "dbscanEps");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*dbscanDensityPtr==NULL)
      {
         if (icComm->commRank()==0) {
            printf("dbscanDensity was not defined in %s; setting to 1\n", CONFIG_FILE);
         }
         *dbscanDensityPtr= strdup("1.0");
      }
      else
      {
         status = checkOctaveArgumentNumeric(*dbscanDensityPtr, "dbscanDensity");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*heatMapMontageDirPtr==NULL)
      {
         fprintf(stderr, "heatMapMontageDir was not defined in %s.\n", CONFIG_FILE);
         status = PV_FAILURE;
      }
      else
      {
         status = checkOctaveArgumentString(*heatMapMontageDirPtr, "heatMapMontageDir");
      }
   }

   if (status==PV_SUCCESS)
   {
      if (*displayCommandPtr==NULL)
      {
         if (icComm->commRank()==0) {
            printf("displayCommand was not defined in %s; leaving blank\n", CONFIG_FILE);
         }
         *displayCommandPtr = strdup("");
      }
      else
      {
         status = checkOctaveArgumentString(*displayCommandPtr, "displayCommand");
      }
   }

   if (icComm->commRank()==0) {
      fclose(parseConfigFileFP);
   }
   return status;
}

int checkOctaveArgumentString(char const * argString, char const * argName)
{
   int status = PV_SUCCESS;
   for (size_t c=0; c<strlen(argString); c++)
   {
      if (argString[c] == '"' or argString[c] == '\'')
      {
         fprintf(stderr, "%s cannot contain quotation marks (\") or apostrophes (')", argName);
         status = PV_FAILURE;
      }
   }
   return status;
}

int checkOctaveArgumentNumeric(char const * argString, char const * argName)
{
   char * endptr = NULL;
   strtod(argString, &endptr);
   int status = PV_SUCCESS;
   if (*endptr!='\0')
   {
      fprintf(stderr, "%s contains characters that do not interpret as numeric.\n", argName);
      status = PV_FAILURE;
   }
   return status;
}

int checkOctaveArgumentVector(char const * argString, char const * argName)
{
   // make sure that the string contains only characters in the set '0123456789[];:, -', and
   // that any comma is preceded by more opening brackets than closing brackets.
   // not perfect, or even very good, but I think it means that the only allowable strings
   // either parse in octave to an array of integers (possibly a vector or a scalar),
   // or causes an immediate error in octave.
   int status = PV_SUCCESS;
   char const * allowable = "0123456789[];:, -";
   int nestingLevel = 0;
   for (char const * s = argString; *s; s++)
   {
      bool allowed = false;
      for (char const * a = allowable; *a; a++)
      {
         if (*s==*a)
         {
            allowed = true;
            break;
         }
      }
      if (!allowed)
      {
         fprintf(stderr, "Only allowable characters in %s are \"%s\"\n", argName, allowable);
         status = PV_FAILURE;
         break;
      }
      if (*s=='[') { nestingLevel++; }
      if (*s==']') { nestingLevel--; }
      if (*s==',' && nestingLevel <= 0)
      {
         fprintf(stderr, "%s cannot have a comma outside of brackets\n", argName);
         status = PV_FAILURE;
         break;
      }
   }
   return status;
}

int parseConfigParameter(PV::InterColComm * icComm, char const * configParameter, char const * configValue, char ** parameterPtr, unsigned int lineNumber)
{
   if (*parameterPtr != NULL)
   {
      fprintf(stderr, "Line %u: Multiple lines defining %s: already set to \"%s\"; duplicate value is \"%s\"\n", lineNumber, configParameter, *parameterPtr, configValue);
      return PV_FAILURE;
   }
   *parameterPtr = strdup(configValue);
   if (*parameterPtr == NULL)
   {
      fprintf(stderr, "Error setting %s from config file: %s\n", configParameter, strerror(errno));
      return PV_FAILURE;
   }
   if (icComm->commRank()==0)
   {
      printf("%s set to \"%s\"\n", configParameter, configValue);
   }
   return PV_SUCCESS;
}

