/*
 * ProbeStatsFlagTest.cpp
 *
 */

#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>

#include <cmath>

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
      float mintolerance  = 1e-6f;
      float maxtolerance  = 1e-6f;
      float meantolerance = 1e-6f;
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
         if (probet != checkt) {
            ErrorLog().printf(
                  "Line %d time value %f does not match expected value %f.\n",
                  linenumber,
                  (double)probet,
                  (double)checkt);
            status = PV_FAILURE;
         }
         if (std::fabs(probemin - checkmin) > mintolerance) {
            ErrorLog().printf(
                  "Line %d min value %f does not match expected value %f.\n",
                  linenumber,
                  (double)probemin,
                  (double)checkmin);
            status = PV_FAILURE;
         }
         if (std::fabs(probemax - checkmax) > maxtolerance) {
            ErrorLog().printf(
                  "Line %d max value %f does not match expected value %f.\n",
                  linenumber,
                  (double)probemax,
                  (double)checkmax);
            status = PV_FAILURE;
         }
         if (std::fabs(probemean - checkmean) > meantolerance) {
            ErrorLog().printf(
                  "Line %d mean value %f does not match expected value %f.\n",
                  linenumber,
                  (double)probemean,
                  (double)checkmean);
            status = PV_FAILURE;
         }
      }
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
