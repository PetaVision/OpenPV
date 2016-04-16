/*
 * PV_KeywordHandler.cpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include "PV_KeywordHandler.hpp"
#include <include/pv_common.h>

namespace PV {

PV_KeywordHandler::PV_KeywordHandler(char const * kw, ObjectCreateFn creator) {
   initialize(kw, creator);
}

PV_KeywordHandler::PV_KeywordHandler(PV_KeywordHandler const& orig) {
   initialize(orig.getKeyword(), orig.getCreator());
}

PV_KeywordHandler& PV_KeywordHandler::operator=(PV_KeywordHandler const& orig) {
   free(keyword);
   initialize(orig.getKeyword(), orig.getCreator());
}

int PV_KeywordHandler::initialize(char const * kw, ObjectCreateFn creator) {
   keyword = strdup(kw);
   if (keyword == NULL) {
      fprintf(stderr, "PV_KeywordHandler unable to store type \"%s\": %s\n", keyword, strerror(errno));
   }
   this->creator = creator;
   return PV_SUCCESS;
}

BasePVObject * PV_KeywordHandler::create(char const * name, HyPerCol * hc) const {
   return (creator)(name, hc);
}

PV_KeywordHandler::~PV_KeywordHandler() {
   free(keyword);
}

} /* namespace PV */
