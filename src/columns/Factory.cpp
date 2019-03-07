/*
 * Factory.cpp
 *
 *  Created on: Jan 20, 2016
 *      Author: pschultz
 */

#include "Factory.hpp"
#include "include/pv_common.h"

namespace PV {

Factory::Factory() {}

// Factory::registerCoreKeywords has been moved to PV::registerCoreKeywords in CoreKeywords.cpp
// The core keywords are no longer automatically added when instantiating the factory.
// Instantiating PV_Init will call registerCoreKeywords().

int Factory::registerKeyword(char const *keyword, ObjectCreateFn creator) {
   KeywordHandler const *keywordHandler = getKeywordHandler(keyword);
   if (keywordHandler != nullptr) {
      return PV_FAILURE;
   }
   KeywordHandler *newKeyword = new KeywordHandler(keyword, creator);
   mKeywordHandlerList.push_back(newKeyword);
   return PV_SUCCESS;
}

BaseObject *Factory::createByKeyword(
      char const *keyword,
      char const *name,
      PVParams *params,
      Communicator *comm) const {
   if (keyword == nullptr) {
      return nullptr;
   }
   KeywordHandler const *keywordHandler = getKeywordHandler(keyword);
   if (keywordHandler == nullptr) {
      auto errorString = std::string(keyword).append(" \"").append(name).append("\": ");
      errorString.append("keyword \"").append(keyword).append("\" is unrecognized.");
      throw std::invalid_argument(errorString);
   }
   return keywordHandler ? keywordHandler->create(name, params, comm) : nullptr;
}

KeywordHandler const *Factory::getKeywordHandler(char const *keyword) const {
   pvAssert(keyword != nullptr);
   for (auto &typeCreator : mKeywordHandlerList) {
      if (!strcmp(typeCreator->getKeyword(), keyword)) {
         return typeCreator;
      }
   }
   return nullptr;
}

int Factory::clearKeywordHandlerList() {
   for (auto &kh : mKeywordHandlerList) {
      delete kh;
   }
   mKeywordHandlerList.clear();
   return PV_SUCCESS;
}

Factory::~Factory() { clearKeywordHandlerList(); }

} /* namespace PV */
