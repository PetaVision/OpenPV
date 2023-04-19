#include "include/pv_common.h"
#include "io/io.hpp"
#include "utils/PathComponents.hpp"
#include "utils/PVLog.hpp"
#include <map>
#include <string>

int compareStrings(
      std::string const &argument,
      std::string const &correct,
      std::string const &observed,
      char const *funcName,
      int prevStatus);
int runTest(std::map<std::string, std::string> const &testParams);

int main(int argc, char *argv[]) {
   std::shared_ptr<PV::Arguments> arguments = PV::parse_arguments(argc, argv, false);
   std::string logFile                      = arguments->getStringArgument("LogFile");
   PV::setLogFile(logFile);

   int status = PV_SUCCESS;
   std::map<std::string, std::string> testparams;

   testparams.clear();
   testparams.emplace("argument", "example");
   testparams.emplace("dirname", ".");
   testparams.emplace("basename", "example");
   testparams.emplace("extension", "");
   testparams.emplace("stripExt", "example");
   status = runTest(testparams) == PV_SUCCESS ? status : PV_FAILURE;

   testparams.clear();
   testparams.emplace("argument", "example.txt");
   testparams.emplace("dirname", ".");
   testparams.emplace("basename", "example.txt");
   testparams.emplace("extension", ".txt");
   testparams.emplace("stripExt", "example");
   status = runTest(testparams) == PV_SUCCESS ? status : PV_FAILURE;

   testparams.clear();
   testparams.emplace("argument", "dir/example.txt");
   testparams.emplace("dirname", "dir");
   testparams.emplace("basename", "example.txt");
   testparams.emplace("extension", ".txt");
   testparams.emplace("stripExt", "example");
   status = runTest(testparams) == PV_SUCCESS ? status : PV_FAILURE;

   testparams.clear();
   testparams.emplace("argument", "top/dir/example.txt");
   testparams.emplace("dirname", "top/dir");
   testparams.emplace("basename", "example.txt");
   testparams.emplace("extension", ".txt");
   testparams.emplace("stripExt", "example");
   status = runTest(testparams) == PV_SUCCESS ? status : PV_FAILURE;

   testparams.clear();
   testparams.emplace("argument", "top/dir.ectory/example");
   testparams.emplace("dirname", "top/dir.ectory");
   testparams.emplace("basename", "example");
   testparams.emplace("extension", "");
   testparams.emplace("stripExt", "example");
   status = runTest(testparams) == PV_SUCCESS ? status : PV_FAILURE;

   if (status == PV_SUCCESS) {
       InfoLog().printf("Test passed.\n");
   }
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int compareStrings(
      std::string const &argument,
      std::string const &correct,
      std::string const &observed,
      char const *funcName,
      int prevStatus) {
   if (correct != observed) {
      ErrorLog().printf(
            "%s failed for \"%s\": expected \"%s\", observed \"%s\"\n",
            funcName, argument.c_str(), correct.c_str(), observed.c_str());
      return PV_FAILURE;
   }
   return prevStatus;
}

int runTest(std::map<std::string, std::string> const &testParams) {
   std::map<std::string, std::string>::const_iterator iter;

   iter = testParams.find("argument");
   FatalIf(iter == testParams.end(), "runTest could not find argument.\n");
   auto argument = iter->second;

   int status = PV_SUCCESS;
   std::string correct, observed;

   iter = testParams.find("dirname");
   FatalIf(iter == testParams.end(), "runTest could not find dirname.\n");
   correct = iter->second;
   observed = PV::dirName(argument);
   status = compareStrings(argument, correct, observed, "dirName", status);

   iter = testParams.find("basename");
   FatalIf(iter == testParams.end(), "runTest could not find basename.\n");
   correct = iter->second;
   observed = PV::baseName(argument);
   status = compareStrings(argument, correct, observed, "baseName", status);

   iter = testParams.find("extension");
   FatalIf(iter == testParams.end(), "runTest could not find extension.\n");
   correct = iter->second;
   observed = PV::extension(argument);
   status = compareStrings(argument, correct, observed, "extension", status);

   iter = testParams.find("stripExt");
   FatalIf(iter == testParams.end(), "runTest could not find stripExt.\n");
   correct = iter->second;
   observed = PV::stripExtension(argument);
   status = compareStrings(argument, correct, observed, "stripExtension", status);

   return status;
}
