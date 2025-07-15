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
      std::shared_ptr<ParamsIO> paramsIO,
      Communicator const *comm) const {
   if (keyword == nullptr) {
      return nullptr;
   }
   KeywordHandler const *keywordHandler = getKeywordHandler(keyword);
   if (keywordHandler == nullptr) {
      std::string const &name = paramsIO->getName();
      auto errorString = std::string(keyword).append(" \"").append(name).append("\": ");
      errorString.append("keyword \"").append(keyword).append("\" is unrecognized.");
      throw std::invalid_argument(errorString);
   }
   return keywordHandler->create(paramsIO, comm);
}

BaseObject *Factory::createByKeyword(
      char const *keyword,
      std::shared_ptr<ParamGroup> params,
      std::shared_ptr<ParamGroup> defaults,
      Communicator const *comm) const {
   if (params == nullptr) {
      return nullptr;
   }
   auto paramsIO = std::make_shared<ParamsIO>(params, defaults);
   return createByKeyword(keyword, paramsIO, comm);
}

BaseObject *Factory::createByKeyword(char const *keyword, BaseObject *baseObject) const {
   BaseObject *newobject = nullptr;
   try {
      auto const *name = baseObject->getName();
      auto paramsIO    = baseObject->getParamsIO();
      auto const *comm = baseObject->getCommunicator();
      newobject        = createByKeyword(keyword, paramsIO, comm);
   } catch (const std::exception &e) {
      Fatal().printf(
            "%s unable to create %s: %s\n", baseObject->getDescription_c(), keyword, e.what());
   }
   FatalIf(
         newobject == nullptr, // Because of try/catch above, this should never happen.
         "%s attempt to create %s returned null pointer.\n",
         baseObject->getDescription_c(),
         keyword);
   return newobject;
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
