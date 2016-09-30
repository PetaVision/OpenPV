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
   free(keyword);
   initialize(orig.getKeyword(), orig.getCreator());
   return *this;
}

int KeywordHandler::initialize(char const *kw, ObjectCreateFn creator) {
   keyword = strdup(kw);
   if (keyword == NULL) {
      pvError().printf(
            "KeywordHandler unable to store type \"%s\": %s\n", keyword, strerror(errno));
   }
   this->creator = creator;
   return PV_SUCCESS;
}

BaseObject *KeywordHandler::create(char const *name, HyPerCol *hc) const {
   return (creator)(name, hc);
}

KeywordHandler::~KeywordHandler() { free(keyword); }

} /* namespace PV */
