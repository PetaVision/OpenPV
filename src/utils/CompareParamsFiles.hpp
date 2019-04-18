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

/**
 * Compares two params files, printing error messages describing any differences.
 * The Communicator argument is needed by the PVParams constructor.
 * Return value is PV_SUCCESS if the params files are equivalent, and PV_FAILURE if not.
 */
int compareParamsFiles(
      std::string const &paramsFile1,
      std::string const &paramsFile2,
      Communicator *comm);

/**
 * Compares two ParameterGroup objects, printing error messages describing any differences.
 * Return value is PV_SUCCESS if the groups are equivalent, and PV_FAILURE if not.
 */
int compareParameterGroups(ParameterGroup *group1, ParameterGroup *group2);

/**
 * Compares two ParameterStack objects, printing error messages describing any differences.
 * The groupName argument is used in any error messages, since ParameterStack does not
 * have a name data member.
 * Return value is PV_SUCCESS if the params files are equivalent, and PV_FAILURE if not.
 */
int compareParameterNumericStacks(
      char const *groupName,
      ParameterStack *stack1,
      ParameterStack *stack2);

/**
 * Compares two ParameterArrayStack objects, printing error messages describing any differences.
 * The groupName argument is used in any error messages, since ParameterArrayStack does not
 * have a name data member.
 * Return value is PV_SUCCESS if the params files are equivalent, and PV_FAILURE if not.
 */
int compareParameterArrayStacks(
      char const *groupName,
      ParameterArrayStack *stack1,
      ParameterArrayStack *stack2);

/**
 * Compares two ParameterArray objects, printing error messages describing any differences.
 * The groupName argument is used in any error messages, since ParameterArray does not
 * have a name data member.
 * Return value is PV_SUCCESS if the params files are equivalent, and PV_FAILURE if not.
 */
int compareParameterArray(char const *groupName, ParameterArray *array1, ParameterArray *array2);

/**
 * Compares two ParameterStringStack objects, printing error messages describing any differences.
 * The groupName argument is used in any error messages, since ParameterStringStack does not
 * have a name data member.
 * Return value is PV_SUCCESS if the params files are equivalent, and PV_FAILURE if not.
 */
int compareParameterStringStacks(
      char const *groupName,
      ParameterStringStack *stack1,
      ParameterStringStack *stack2);

/**
 * Compares two ParameterString objects, printing error messages describing any differences.
 * The groupName argument is used in any error messages, since ParameterString does not
 * have a name data member.
 * Return value is PV_SUCCESS if the params files are equivalent, and PV_FAILURE if not.
 */
int compareParameterString(
      char const *groupName,
      ParameterString *string1,
      ParameterString *string2);

} // end namespace PV

#endif // COMPAREPARAMSFILES_HPP_
