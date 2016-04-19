/*
 * KeywordHandler.cpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include "KeywordHandler.hpp"
#include <include/pv_common.h>

namespace PV {

KeywordHandler::KeywordHandler(char const * kw, ObjectCreateFn creator) {
   initialize(kw, creator);
}

KeywordHandler::KeywordHandler(KeywordHandler const& orig) {
   initialize(orig.getKeyword(), orig.getCreator());
}

KeywordHandler& KeywordHandler::operator=(KeywordHandler const& orig) {
   free(keyword);
   initialize(orig.getKeyword(), orig.getCreator());
}

int KeywordHandler::initialize(char const * kw, ObjectCreateFn creator) {
   keyword = strdup(kw);
   if (keyword == NULL) {
      fprintf(stderr, "KeywordHandler unable to store type \"%s\": %s\n", keyword, strerror(errno));
   }
   this->creator = creator;
   return PV_SUCCESS;
}

BaseObject * KeywordHandler::create(char const * name, HyPerCol * hc) const {
   return (creator)(name, hc);
}

KeywordHandler::~KeywordHandler() {
   free(keyword);
}

} /* namespace PV */
