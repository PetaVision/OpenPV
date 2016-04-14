/*
 * buildandrun.hpp
 *
 *  Created on: May 27, 2011
 *      Author: peteschultz
 */

#ifndef BUILDANDRUN_HPP_
#define BUILDANDRUN_HPP_

#include <time.h>
#include <string>
#include <iostream>

#include "../include/pv_common.h"

#include "../columns/HyPerCol.hpp"

#include "../weightinit/InitWeights.hpp"
#include "../normalizers/NormalizeBase.hpp"
#include "PV_Init.hpp"


#include "../io/ParamGroupHandler.hpp"
#include "../io/CoreParamGroupHandler.hpp"

using namespace PV;




// The build, buildandrun1paramset, and buildandrun functions are included for backwards compatibility.  The three versions after them, which use ParamGroupHandler arguments, are preferred.
int buildandrun(int argc, char * argv[],
                int (*custominit)(HyPerCol *, int, char **) = NULL,
                int (*customexit)(HyPerCol *, int, char **) = NULL,
                void * (*customgroups)(const char *, const char *, HyPerCol *) = NULL);
int rebuildandrun(PV_Init * initObj,
                int (*custominit)(HyPerCol *, int, char **) = NULL,
                int (*customexit)(HyPerCol *, int, char **) = NULL,
                void * (*customgroups)(const char *, const char *, HyPerCol *) = NULL);

int buildandrun1paramset(PV_Init* initObj,
                         int (*custominit)(HyPerCol *, int, char **),
                         int (*customexit)(HyPerCol *, int, char **),
                         void * (*customgroups)(const char *, const char *, HyPerCol *));

HyPerCol * build(PV_Init* initObj,
      void * (*customgroups)(const char *, const char *, HyPerCol *) = NULL);

// The build, buildandrun1paramset, and buildandrun functions below are preferred to the versions above.
int buildandrun(int argc, char * argv[],
                int (*custominit)(HyPerCol *, int, char **),
                int (*customexit)(HyPerCol *, int, char **),
                ParamGroupHandler ** groupHandlerList, int numGroupHandlers);

int rebuildandrun(PV_Init * initObj,
                int (*custominit)(HyPerCol *, int, char **),
                int (*customexit)(HyPerCol *, int, char **),
                ParamGroupHandler ** groupHandlerList, int numGroupHandlers);

int buildandrun1paramset(PV_Init* initObj,
                         int (*custominit)(HyPerCol *, int, char **),
                         int (*customexit)(HyPerCol *, int, char **),
                         ParamGroupHandler ** groupHandlerList, int numGroupHandlers);

// Deprecated April 14, 2016, in favor of using the dryRunFlag in PV_Arguments and the -n flag on the command line.
int outputParams(int argc, char * argv[], char const * path, ParamGroupHandler ** groupHandlerList, int numGroupHandlers);

HyPerCol * build(PV_Init * initObj,
                 ParamGroupHandler ** groupHandlerList,
                 int numGroupHandlers);

ParamGroupHandler * getGroupHandlerFromList(char const * keyword, CoreParamGroupHandler * coreHandler, ParamGroupHandler ** groupHandlerList, int numGroupHandlers, ParamGroupType * foundGroupType);
BaseConnection * createConnection(CoreParamGroupHandler * coreGroupHandler, ParamGroupHandler ** customHandlerList, int numGroupHandlers, char const * keyword, char const * groupname, HyPerCol * hc);

int checknewobject(void * object, const char * kw, const char * name, HyPerCol * hc);

#endif /* BUILDANDRUN_HPP_ */
