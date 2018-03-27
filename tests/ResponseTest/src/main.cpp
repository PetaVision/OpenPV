#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <map>
#include <observerpattern/Response.hpp>
#include <utils/PVLog.hpp>
#include <vector>

typedef PV::Response::Status Status;

struct AnswerKey {
   std::map<Status, std::size_t> lookup;
   std::vector<Status> input;
   std::vector<std::string> description;
   std::vector<std::vector<Status>> correctStatus;
   std::vector<std::vector<std::string>> correctDescription;

   AnswerKey();
};

AnswerKey::AnswerKey() {
   Status const success  = PV::Response::SUCCESS;
   Status const noAction = PV::Response::NO_ACTION;
   Status const partial  = PV::Response::PARTIAL;
   Status const postpone = PV::Response::POSTPONE;

   input       = {success, noAction, partial, postpone};
   description = {"Success", "No Action", "Partial", "Postpone"};

   std::size_t const N = input.size();

   lookup.clear();
   for (std::size_t n = 0; n < N; n++) {
      lookup.insert({input[n], n});
   }

   correctStatus.resize(N);
   correctStatus[0] = {success, success, partial, partial};
   correctStatus[1] = {success, noAction, partial, postpone};
   correctStatus[2] = {partial, partial, partial, partial};
   correctStatus[3] = {partial, postpone, partial, postpone};

   correctDescription.resize(N);
   for (std::size_t m = 0; m < N; m++) {
      correctDescription[m].resize(N);
      for (std::size_t n = 0; n < N; n++) {
         std::size_t index        = lookup.find(correctStatus[m][n])->second;
         correctDescription[m][n] = description[index];
      }
   }
}

int checkResult(Status const &x, Status const &y, Status const &z, AnswerKey const &cs);

int main(int argc, char *argv[]) {
   auto *pvInit = new PV::PV_Init(&argc, &argv, false /*no unrecognized arguments allowed*/);

   int status = PV_SUCCESS;

   AnswerKey cs;

   std::size_t const N = cs.input.size();

   // Test operator+ for Response::Status arguments
   for (std::size_t m = 0; m < N; m++) {
      for (std::size_t n = 0; n < N; n++) {
         auto x = cs.input[m];
         auto y = cs.input[n];
         auto z = x + y;
         if (checkResult(x, y, z, cs) != PV_SUCCESS) {
            status = PV_FAILURE;
         }
      }
   }
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }

   InfoLog() << "Test passed.\n";
   delete pvInit;
   return EXIT_SUCCESS;
}

int checkResult(Status const &x, Status const &y, Status const &z, AnswerKey const &cs) {
   // Check if z is what the sum of x and y should be.
   std::size_t m = cs.lookup.find(x)->second;
   std::size_t n = cs.lookup.find(y)->second;
   std::size_t p = cs.lookup.find(z)->second;
   if (z != cs.correctStatus[x][y]) {
      ErrorLog().printf(
            "Combining %s and %s should give %s; instead it gave %s.\n",
            cs.description[m].c_str(),
            cs.description[n].c_str(),
            cs.correctDescription[m][n].c_str(),
            cs.description[p].c_str());
      return PV_FAILURE;
   }
   return PV_SUCCESS;
}
