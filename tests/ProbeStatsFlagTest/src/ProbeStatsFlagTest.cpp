/*
 * ProbeStatsFlagTest.cpp
 *
 */

#include "CheckValue.hpp"

#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>

#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

char const *gProbeFile = "output/OutputStats.txt";

int main(int argc, char *argv[]) {
   PV::PV_Init pv_initObj(&argc, &argv, false /*do not allow unrecognized arguments*/);
   if (pv_initObj.getCommunicator()->commRank() == 0) {
      struct stat probeFileStatBuf;
      int statstatus = stat(gProbeFile, &probeFileStatBuf);
      if (!statstatus) {
         int unlinkstatus = unlink(gProbeFile);
         FatalIf(unlinkstatus, "Unable to delete previously existing file \"%s\"\n", gProbeFile);
      }
   }

   int status = buildandrun(&pv_initObj);
   if (status != PV_SUCCESS) {
      return EXIT_FAILURE;
   }

   if (pv_initObj.getCommunicator()->commRank() == 0) {
      char const *checkFile = "input/correctOutputStats.txt";
      FILE *probefp         = std::fopen(gProbeFile, "r");
      FatalIf(!probefp, "Unable to open probe output \"%s\": %s\n", gProbeFile, strerror(errno));
      FILE *checkfp = std::fopen(checkFile, "r");
      FatalIf(!checkfp, "Unable to open checkfile \"%s\": %s\n", checkFile, strerror(errno));

      int linenumber      = 0;
      float mintolerance  = 2e-7f;
      float maxtolerance  = 2e-7f;
      float meantolerance = 2e-7f;
      while (true) {
         float probet, checkt, probemin, checkmin, probemax, checkmax, probemean, checkmean;

         linenumber++;
         std::string linenumberString(std::string("Line ") + std::to_string(linenumber));
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
                  "Line %d of \"%s\" does not match \"%s\".\n", linenumber, gProbeFile, checkFile);
            break;
         }
         if (probenumread == EOF) {
            break;
         }

         if (probenumread != 4) {
            ErrorLog().printf(
                  "Line %d of \"%s\" does not have the expected format.\n", linenumber, gProbeFile);
            status = PV_FAILURE;
            break;
         }

         try {
            checkValue(linenumberString, std::string("time"), probet, checkt, 0.0f);
         } catch (std::exception const &e) {
            ErrorLog() << e.what();
            status = PV_FAILURE;
         }

         try {
            checkValue(linenumberString, std::string("min"), probemin, checkmin, mintolerance);
         } catch (std::exception const &e) {
            ErrorLog() << e.what();
            status = PV_FAILURE;
         }

         try {
            checkValue(linenumberString, std::string("max"), probemax, checkmax, maxtolerance);
         } catch (std::exception const &e) {
            ErrorLog() << e.what();
            status = PV_FAILURE;
         }

         try {
            checkValue(linenumberString, std::string("mean"), probemean, checkmean, meantolerance);
         } catch (std::exception const &e) {
            ErrorLog() << e.what();
            status = PV_FAILURE;
         }
      }
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
