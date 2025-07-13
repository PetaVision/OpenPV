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
#include "params/PVParams.hpp"

namespace PV {

/**
 * Compares two params files, printing error messages describing any differences.
 * The Communicator argument is needed by the PVParams constructor.
 * Return value is PV_SUCCESS if the params files are equivalent, and PV_FAILURE if not.
 */
int compareParamsFiles(
      std::string const &paramsFile1,
      std::string const &paramsFile2,
      MPI_Comm mpiComm);

/**
 * Compares two ParamGroup objects, printing error messages describing any differences.
 * Return value is PV_SUCCESS if the groups are equivalent, and PV_FAILURE if not.
 */
int compareParamGroups(std::shared_ptr<ParamGroup> group1, std::shared_ptr<ParamGroup> group2);

} // end namespace PV

#endif // COMPAREPARAMSFILES_HPP_
