/*
 * TauswortheURBGTest.cpp
 *
 */

#include <utils/TauswortheURBG.hpp>

#include <cstdlib>
#include <vector>

int main(int argc, char *argv[]) {
   std::vector<unsigned int> expected{
         524493632U,
         2205310259U,
         584444467U,
         3572813119U,
         3669225849U,
         1873049212U,
         3854295868U,
         2523728678U};
   unsigned int baseSeed = 2658309590U;
   std::vector<unsigned int> v(8);
   for (unsigned int k = 0; k < 8; ++k) {
      PV::TauswortheURBG rng(baseSeed + k);
      unsigned int result;
      rng();
      rng();
      rng();
      result = rng();
      v[k] = result;
   }
   return v == expected ? EXIT_SUCCESS : EXIT_FAILURE;
}
