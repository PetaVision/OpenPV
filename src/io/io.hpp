/*
 * io.hpp
 *
 *  Created on: Oct 24, 2008
 *      Author: rasmussn
 */

#ifndef IO_HPP_
#define IO_HPP_

#include "columns/Arguments.hpp" // Return value of parse_arguments
#include <memory> // shared_ptr returned by parse_arguments

namespace PV {

int pv_getopt(int argc, char const *const *argv, const char *opt, bool *paramusage);
int pv_getopt_int(int argc, char const *const *argv, const char *opt, int *iVal, bool *paramusage);
int pv_getoptionalopt_int(
      int argc,
      char const *const *argv,
      const char *opt,
      int *iVal,
      bool *defaultVal,
      bool *paramusage);
int pv_getopt_str(
      int argc,
      char const *const *argv,
      const char *opt,
      char **sVal,
      bool *paramusage);
int pv_getopt_long(
      int argc,
      char const *const *argv,
      const char *opt,
      long int *ulVal,
      bool *paramusage);
int pv_getopt_unsigned(
      int argc,
      char const *const *argv,
      const char *opt,
      unsigned int *uVal,
      bool *paramusage);

std::shared_ptr<Arguments> parse_arguments(
      int argc, char *argv[], bool allowUnrecognizedArguments);

int parse_options(
      int argc,
      char const *const *argv,
      bool *paramusage,
      bool *require_return,
      char **output_path,
      char **param_file,
      char **log_file,
      char **gpu_devices,
      unsigned int *random_seed,
      char **working_dir,
      int *restart,
      char **checkpointReadDir,
      bool *useDefaultNumThreads,
      int *numthreads,
      int *numRows,
      int *numColumns,
      int *batch_width,
      int *dryrun,
      unsigned int *shuffleParamGroups);

} // end namespace PV

#endif /* IO_HPP_ */
