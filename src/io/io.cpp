/*
 * io.c
 *
 *  Created on: Aug 3, 2008
 *      Author: dcoates
 */

// Shared input and output routines

#include "io.hpp"

#include <assert.h>
#include <float.h> // FLT_MAX/MIN
#include <math.h>
#include <string.h> // memcpy
#include <string>

#include "utils/PVLog.hpp"

namespace PV {

void usage() {
   pvInfo().printf("\nUsage:\n");
   pvInfo().printf(" -p <parameters filename>\n");
   pvInfo().printf(" [-o <output directory>\n");
   pvInfo().printf(" [-s <random number generator seed>]\n");
   pvInfo().printf(" [-d [<GPU device>,<GPU device>,...]]\n");
   pvInfo().printf(" [-l <output log file>]\n");
   pvInfo().printf(" [-w <working directory>]\n");
   pvInfo().printf(" [-r|-c <checkpoint directory>]\n");
#ifdef PV_USE_OPENMP_THREADS
   pvInfo().printf(" [-t [number of threads]\n");
   pvInfo().printf(" [-n]\n");
#endif // PV_USE_OPENMP_THREADS
}

/**
 * @argc
 * @argv
 * @input_file
 * @param_file
 * @n_time_steps
 * @device
 */
int parse_options(
      int argc,
      char *argv[],
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
      int *num_rows,
      int *num_columns,
      int *batch_width,
      int *dry_run) {
   paramusage[0] = true;
   int arg;
   for (arg = 1; arg < argc; arg++) {
      paramusage[arg] = false;
   }

   if (pv_getopt(argc, argv, "--require-return", paramusage) == 0) {
      *require_return = true;
   }
   pv_getopt_str(argc, argv, "-d", gpu_devices, paramusage);
   pv_getoptionalopt_int(argc, argv, "-t", numthreads, useDefaultNumThreads, paramusage);
   pv_getopt_str(argc, argv, "-o", output_path, paramusage);
   pv_getopt_str(argc, argv, "-p", param_file, paramusage);
   pv_getopt_str(argc, argv, "-l", log_file, paramusage);
   pv_getopt_unsigned(argc, argv, "-s", random_seed, paramusage);
   pv_getopt_str(argc, argv, "-w", working_dir, paramusage);
   if (pv_getopt(argc, argv, "-r", paramusage) == 0) {
      *restart = 1;
   }
   pv_getopt_str(argc, argv, "-c", checkpointReadDir, paramusage);
   pv_getopt_int(argc, argv, "-rows", num_rows, paramusage);
   pv_getopt_int(argc, argv, "-columns", num_columns, paramusage);
   pv_getopt_int(argc, argv, "-batchwidth", batch_width, paramusage);
   if (pv_getopt(argc, argv, "-n", paramusage) == 0) {
      *dry_run = 1;
   }

   return 0;
}

/*
 * @argc
 * @argv
 * @opt
 */
int pv_getopt(int argc, char *argv[], const char *opt, bool *paramusage) {
   int i;
   for (i = 1; i < argc; i++) {
      if (strcmp(argv[i], opt) == 0) {
         if (paramusage) {
            paramusage[i] = true;
         }
         return 0;
      }
   }
   return -1; // not found
}

/**
 * @argc
 * @argv
 * @opt
 * @iVal
 */
int pv_getopt_int(int argc, char *argv[], const char *opt, int *iVal, bool *paramusage) {
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i + 1 < argc && strcmp(argv[i], opt) == 0) {
         if (iVal != NULL)
            *iVal = atoi(argv[i + 1]);
         if (paramusage) {
            paramusage[i]     = true;
            paramusage[i + 1] = true;
         }
         return 0;
      }
   }
   return -1; // not found
}

int pv_getoptionalopt_int(
      int argc,
      char *argv[],
      const char *opt,
      int *iVal,
      bool *useDefaultVal,
      bool *paramusage) {
   int i;
   for (i = 1; i < argc; i += 1) {
      if (strcmp(argv[i], opt) == 0) {
         // Default parameter
         if (i + 1 >= argc || argv[i + 1][0] == '-') {
            if (iVal != NULL) {
               *iVal = -1;
            }
            if (useDefaultVal != NULL) {
               *useDefaultVal = true;
            }
            if (paramusage) {
               paramusage[i] = true;
            }
         }
         else {
            if (iVal != NULL) {
               *iVal = atoi(argv[i + 1]);
            }
            if (useDefaultVal != NULL) {
               *useDefaultVal = false;
            }
            if (paramusage) {
               paramusage[i]     = true;
               paramusage[i + 1] = true;
            }
         }
         return 0;
      }
   }
   if (iVal != NULL) {
      *iVal = -1;
   }
   if (useDefaultVal != NULL) {
      *useDefaultVal = false;
   }
   return -1; // not found
}

/**
 * @argc
 * @argv
 * @opt
 * @iVal
 */
int pv_getopt_long(int argc, char *argv[], const char *opt, long int *iVal, bool *paramusage) {
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i + 1 < argc && strcmp(argv[i], opt) == 0) {
         if (iVal != NULL)
            *iVal = strtol(argv[i + 1], NULL, 0);
         if (paramusage) {
            paramusage[i]     = true;
            paramusage[i + 1] = true;
         }
         return 0;
      }
   }
   return -1; // not found
}

/**
 * @argc
 * @argv
 * @opt
 * @iVal
 */
int pv_getopt_unsigned(
      int argc,
      char *argv[],
      const char *opt,
      unsigned int *uVal,
      bool *paramusage) {
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i + 1 < argc && strcmp(argv[i], opt) == 0) {
         if (uVal != NULL)
            *uVal = (unsigned int)strtoul(argv[i + 1], NULL, 0);
         if (paramusage) {
            paramusage[i]     = true;
            paramusage[i + 1] = true;
         }
         return 0;
      }
   }
   return -1; // not found
}

/**
 * @argc
 * @argv
 * @opt
 * @sVal
 */
int pv_getopt_str(int argc, char *argv[], const char *opt, char **sVal, bool *paramusage) {
   // sVal can be NULL.  If sVal is not null and the option is found,
   // the value of the option is put into sVal and the calling routine is
   // responsible for freeing it.
   // Example: if argv[1] is "-p" and argv[2] is "params.pv", and opt is "-p",
   // *sVal will be "params.pv".
   int i;
   for (i = 1; i < argc; i += 1) {
      if (i + 1 < argc && strcmp(argv[i], opt) == 0) {
         if (sVal != NULL)
            *sVal = strdup(argv[i + 1]);
         if (paramusage) {
            paramusage[i]     = true;
            paramusage[i + 1] = true;
         }
         return 0;
      }
   }
   if (sVal != NULL)
      *sVal = NULL;
   return -1; // not found
}

/**
 * @V
 * @nx0
 * @ny0
 * @nx
 * @ny
 */
int pv_center_image(float *V, int nx0, int ny0, int nx, int ny) {
   int i0, j0, i, j, ii;

   float *buf = (float *)malloc(nx0 * ny0 * sizeof(float));
   assert(buf != NULL);

   assert(nx0 <= nx);
   assert(ny0 <= ny);

   memcpy(buf, V, nx0 * nx0 * sizeof(float));

   i0 = nx / 2 - nx0 / 2;
   j0 = ny / 2 - ny0 / 2;

   for (i = 0; i < nx * ny; i++) {
      V[i] = 0;
   }

   ii = 0;
   for (j = j0; j < j0 + ny0; j++) {
      for (i = i0; i < i0 + nx0; i++) {
         V[i + nx * j] = buf[ii++];
      }
   }
   free(buf);

   return 0;
}

std::string expandLeadingTilde(char const *path) {
   if (path == NULL) {
      return std::string("");
   }
   char *newpath = NULL;
   if (path != NULL) {
      int len = strlen(path);
      if (len == 1 && path[0] == '~') {
         newpath = strdup(getenv("HOME"));
         if (newpath == NULL) {
            pvError().printf(
                  "Unable to expand \"%s\": "
                  "home directory not defined\n",
                  path);
            exit(EXIT_FAILURE);
         }
      }
      else if (len > 1 && path[0] == '~' && path[1] == '/') {
         char *homedir = getenv("HOME");
         if (homedir == NULL) {
            pvError().printf(
                  "Unable to expand \"%s\": "
                  "home directory not defined\n",
                  path);
         }
         char dummy;
         int chars_needed = snprintf(&dummy, 0, "%s/%s", homedir, &path[2]);
         newpath          = (char *)malloc(chars_needed + 1);
         if (newpath == NULL) {
            pvError().printf("Unable to allocate memory for path \"%s/%s\"\n", homedir, &path[2]);
         }
         int chars_used = snprintf(newpath, chars_needed + 1, "%s/%s", homedir, &path[2]);
         assert(chars_used == chars_needed);
      }
      else {
         newpath = strdup(path);
      }
   }
   pvErrorIf(newpath == NULL, "Could not expand path: %s\n", path);
   std::string result(newpath);
   free(newpath);
   return result;
}

} // namespace PV
