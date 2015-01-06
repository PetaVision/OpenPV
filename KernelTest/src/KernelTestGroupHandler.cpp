/*
 * KernelTestGroupHandler.cpp
 *
 *  Created on: Dec 30, 2014
 *      Author: pschultz
 */

#include "KernelTestGroupHandler.hpp"
#include "KernelTestProbe.hpp"

#include <stddef.h>
#include <stdio.h>

namespace PV {

KernelTestGroupHandler::KernelTestGroupHandler() {
}

KernelTestGroupHandler::~KernelTestGroupHandler() {
}

void * KernelTestGroupHandler::createObject(char const * keyword, char const * name, HyPerCol * hc) {
   int status;
   void * addedGroup = NULL;
   bool errorFound = false;
   if( !strcmp(keyword, "KernelTestProbe") ) {
      addedGroup = (void *) new KernelTestProbe(name, hc);
      if( !addedGroup ) {
         fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         errorFound = true;
      }
   }
   return addedGroup;
}

} /* namespace PV */
