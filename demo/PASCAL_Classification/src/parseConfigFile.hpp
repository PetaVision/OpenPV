/*
 * parseConfigFile.hpp
 *
 *  Created on: Sep 24, 2015
 *      Author: pschultz
 */

#ifndef PARSECONFIGFILE_HPP_
#define PARSECONFIGFILE_HPP_

#include <columns/InterColComm.hpp>

int parseConfigFile(PV::InterColComm * icComm, char ** imageLayerNamePtr, char ** resultLayerNamePtr, char ** resultTextFilePtr, char ** octaveCommandPtr, char ** octaveLogFilePtr, char ** classNamesPtr, char ** evalCategoryIndicesPtr, char ** displayCategoryIndicesPtr, char ** highlightThresholdPtr, char ** heatMapThresholdPtr, char ** heatMapMaximumPtr, char ** drawBoundingBoxesPtr, char ** boundingBoxThicknessPtr, char ** dbscanEpsPtr, char ** dbscanDensityPtr, char ** heatMapMontageDirPtr, char ** displayCommandPtr);
int parseConfigParameter(PV::InterColComm * icComm, char const * inputLine, char const * configParameter, char ** parameterPtr, unsigned int lineNumber);
int checkOctaveArgumentString(char const * argString, char const * argName);
int checkOctaveArgumentNumeric(char const * argString, char const * argName);
int checkOctaveArgumentVector(char const * argString, char const * argName);

#endif /* PARSECONFIGFILE_HPP_ */
