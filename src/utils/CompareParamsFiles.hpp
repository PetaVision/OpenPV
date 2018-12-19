/*
 * CompareParamsFiles.hpp
 *
 *  Created on: Dec 17, 2018
 *      Author: pschultz
 *
 *  A set of utility functions for comparing the contents of two params files,
 *  without regard to the order of parameter groups, or the order of parameters
 *  within a group.
 *  Used by the compareparams tool and the DryRunFlagTest system test.
 *  All functions in this file return PV_SUCCESS if the objects are equivalent,
 *  or PV_FAILURE if the objects differ.
 */

#ifndef COMPAREPARAMSFILES_HPP_
#define COMPAREPARAMSFILES_HPP_

#include "columns/Communicator.hpp"
#include "io/PVParams.hpp"

namespace PV {

int compareParamsFiles(
      std::string const &paramsFile1,
      std::string const &paramsFile2,
      Communicator *comm);
int compareParameterGroups(ParameterGroup *group1, ParameterGroup *group2);
int compareParameterNumericStacks(
      char const *groupName,
      ParameterStack *stack1,
      ParameterStack *stack2);
int compareParameterArrayStacks(
      char const *groupName,
      ParameterArrayStack *stack1,
      ParameterArrayStack *stack2);
int compareParameterArray(char const *groupName, ParameterArray *array1, ParameterArray *array2);
int compareParameterStringStacks(
      char const *groupName,
      ParameterStringStack *stack1,
      ParameterStringStack *stack2);
int compareParameterString(
      char const *groupName,
      ParameterString *string1,
      ParameterString *string2);

} // end namespace PV

#endif // COMPAREPARAMSFILES_HPP_
