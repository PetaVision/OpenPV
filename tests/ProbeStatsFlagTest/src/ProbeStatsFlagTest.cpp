/*
 * ProbeStatsFlagTest.cpp
 *
 */

#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>

#include <cmath>

int checkValue(
      float observed, float expected, float tolerance, int lineno, char const *valueDescription);

int main(int argc, char *argv[]) {
   PV::PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   int status = buildandrun(&pv_initObj);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   if (pv_initObj.getCommunicator()->commRank() == 0) {
      char const *probeFile = "output/OutputStats.txt";
      char const *checkFile = "input/correctOutputStats.txt";
      FILE *probefp         = std::fopen(probeFile, "r");
      FatalIf(!probefp, "Unable to open probe output \"%s\": %s\n", probeFile, strerror(errno));
      FILE *checkfp = std::fopen(checkFile, "r");
      FatalIf(!checkfp, "Unable to open checkfile \"%s\": %s\n", checkFile, strerror(errno));

      int linenumber      = 0;
      float mintolerance  = 2e-7f;
      float maxtolerance  = 2e-7f;
      float meantolerance = 2e-7f;
      while (true) {
         float probet, checkt, probemin, checkmin, probemax, checkmax, probemean, checkmean;

         linenumber++;
         int probenumread = fscanf(
               probefp,
               "t=%f, min=%f, max=%f, mean=%f\n",
               &probet,
               &probemin,
               &probemax,
               &probemean);
         int checknumread = fscanf(
               checkfp,
               "t=%f, min=%f, max=%f, mean=%f\n",
               &checkt,
               &checkmin,
               &checkmax,
               &checkmean);
         if (probenumread != checknumread) {
            ErrorLog().printf(
                  "Line %d of \"%s\" does not match \"%s\".\n", linenumber, probeFile, checkFile);
            break;
         }
         if (probenumread == EOF) {
            break;
         }
         if (probenumread != 4) {
            ErrorLog().printf(
                  "Line %d of \"%s\" does not have the expected format.\n", linenumber, probeFile);
            status = PV_FAILURE;
            break;
         }
         if (checkValue(probet, checkt, 0.0f, linenumber, "time") != PV_SUCCESS) {
            status = PV_FAILURE;
         }
         if (checkValue(probemin, checkmin, mintolerance, linenumber, "min") != PV_SUCCESS) {
            status = PV_FAILURE;
         }
         if (checkValue(probemax, checkmax, maxtolerance, linenumber, "max") != PV_SUCCESS) {
            status = PV_FAILURE;
         }
         if (checkValue(probemean, checkmean, meantolerance, linenumber, "mean") != PV_SUCCESS) {
            status = PV_FAILURE;
         }
      }
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkValue(
      float observed, float expected, float tolerance, int lineno, char const *valueDescription) {
   int status = PV_SUCCESS;
   if (expected) {
      float discrepancy = std::fabs(observed - expected);
      float relError    = discrepancy / relError;
      if (relError > tolerance) {
         ErrorLog().printf(
               "Line %d %s value %f does not match expected value %f (discrepancy %g).\n",
               lineno, valueDescription, (double)observed, (double)expected, (double)discrepancy);
         status = PV_FAILURE;
      }
   }
   else {
      if (observed) {
         ErrorLog().printf(
               "Line %d %s value %g does not match expected value of zero.\n",
               lineno, valueDescription, (double)observed);
         status = PV_FAILURE;
      }
   }
   return status;
}
