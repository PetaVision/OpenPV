/*
 * ExpandLeadingTildeTest main.cpp
 *
 */

#include "columns/PV_Init.hpp"
#include "utils/ExpandLeadingTilde.hpp"
#include "utils/PVLog.hpp"
#include <cstdlib>

void checkTildeExpansion(std::string const &in, std::string const &expected);

int main(int argc, char *argv[]) {
   auto *pv_init = new PV::PV_Init(&argc, &argv, false /*do not allow unrecognized arguments*/);

   if (pv_init->getCommunicator()->globalCommRank() == 0) {
      char const *homeDirArray  = std::getenv("HOME");
      std::string homeDirString = PV::getHomeDirectory();

      FatalIf(
            homeDirString != std::string(homeDirArray),
            "getHomeDirectory returned \"%s\" instead of expected \"%s\"\n",
            homeDirString.c_str(),
            homeDirArray);

      std::string in, expected, observed;

      //////// relative path name should not be affected ////////
      checkTildeExpansion("abc/def/ghi", "abc/def/ghi");

      //////// absolute path name should not be affected ////////
      checkTildeExpansion("/abc/def/ghi", "/abc/def/ghi");

      //////// Lone tilde should be expanded ////////
      checkTildeExpansion("~", homeDirString);

      //////// "~/" should be expanded ////////
      checkTildeExpansion("~/", homeDirString + '/');

      //////// "~/" followed by a relative path should be expanded ////////
      checkTildeExpansion("~/abc/def", homeDirString + "/abc/def");

      //////// "~" followed by a nonslash should not be expanded (TODO for other usernames?)
      checkTildeExpansion("~abc/def", "~abc/def");
   }

   delete pv_init;

   return EXIT_SUCCESS;
}

void checkTildeExpansion(std::string const &input, std::string const &expected) {
   auto observed = PV::expandLeadingTilde(input);
   FatalIf(
         observed != expected,
         "expandLeadingTilde(string(\"%s\")) returned \"%s\" instead of expected \"%s\"\n",
         input.c_str(),
         observed.c_str(),
         expected.c_str());

   observed = PV::expandLeadingTilde(input.c_str());
   FatalIf(
         observed != expected,
         "expandLeadingTilde(char *(\"%s\")) returned \"%s\" instead of expected \"%s\"\n",
         input.c_str(),
         observed.c_str(),
         expected.c_str());
}
