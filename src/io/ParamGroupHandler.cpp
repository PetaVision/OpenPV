/*
 * ParamGroupHandler.cpp
 *
 *  Created on: Dec 30, 2014
 *      Author: pschultz
 */

// Note: ParamGroupHandler and functions that depend on it were deprecated
// on March 24, 2016.  Instead, creating layers, connections, etc. should
// be handled using the PV_Init::registerKeyword, PV_Init::create, and
// PV_Init::build methods.

#include "ParamGroupHandler.hpp"

namespace PV {

ParamGroupHandler::ParamGroupHandler() {
}

ParamGroupHandler::~ParamGroupHandler() {
}

} /* namespace PV */
