/*
 * KeywordHandler.cpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#include "KeywordHandler.hpp"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <include/pv_common.h>

namespace PV {

KeywordHandler::KeywordHandler(char const *kw, ObjectCreateFn creator) { initialize(kw, creator); }

KeywordHandler::KeywordHandler(KeywordHandler const &orig) {
   initialize(orig.getKeyword(), orig.getCreator());
}

KeywordHandler &KeywordHandler::operator=(KeywordHandler const &orig) {
   free(mKeyword);
   initialize(orig.getKeyword(), orig.getCreator());
   return *this;
}

int KeywordHandler::initialize(char const *kw, ObjectCreateFn creator) {
   mKeyword = strdup(kw);
   if (mKeyword == NULL) {
      Fatal().printf("KeywordHandler unable to store type \"%s\": %s\n", mKeyword, strerror(errno));
   }
   mCreator = creator;
   return PV_SUCCESS;
}

BaseObject *
KeywordHandler::create(char const *name, PVParams *params, Communicator const *comm) const {
   return (mCreator)(name, params, comm);
}

KeywordHandler::~KeywordHandler() { free(mKeyword); }

} /* namespace PV */
