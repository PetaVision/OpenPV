#include <arch/mpi/mpi.h>
#include <columns/buildandrun.hpp>
#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <include/pv_common.h>
#include <io/PVParams.hpp>
#include <observerpattern/ObserverTable.hpp>
#include <probes/ColumnEnergyProbe.hpp>
#include <probes/L2NormProbe.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>
#include <memory>
#include <string>

int checkResult(HyPerCol *hc, int argc, char *argv[]);

int main(int argc, char **argv) {
   PV::PV_Init pv_init(&argc, &argv, false);
   int status = buildandrun(&pv_init, nullptr /*custominit*/, checkResult);

   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkResult(HyPerCol *hypercol, int argc, char *argv[]) {
   std::string componentName("LayerProbe");
   std::string columnEnergyProbeName("TestColumnEnergyProbe");

   auto *objectFromColumn = hypercol->getObjectFromName(columnEnergyProbeName);
   PV::ColumnEnergyProbe *columnProbe = dynamic_cast<PV::ColumnEnergyProbe *>(objectFromColumn);
   FatalIf(
         columnProbe == nullptr,
         "No ColumnEnergyProbe \"%s\" in params.\n", columnEnergyProbeName.c_str());

   objectFromColumn = hypercol->getObjectFromName(componentName);
   PV::L2NormProbe *l2normProbe = dynamic_cast<PV::L2NormProbe *>(objectFromColumn);
   FatalIf(
         l2normProbe == nullptr, "No L2NormProbe \"%s\" in params.\n", componentName.c_str());

   double coefficient = l2normProbe->getCoefficient();
   auto l2normValues = l2normProbe->getValues();
   auto columnValues = columnProbe->getValues();

   FatalIf(
         columnValues.size() != l2normValues.size(),
         "Column probe has a different number of values than layer probe: %zu versus %zu\n",
         columnValues.size(), l2normValues.size());

   int status = PV_SUCCESS;
   auto N = columnValues.size();
   for (decltype(N) n = 0; n < N; ++n) {
      if (columnValues[n] != l2normValues[n] * coefficient) {
         ErrorLog().printf("Discrepancy in value %d: column should be %f * %f = %f, but is %f\n",
         n, coefficient, l2normValues[n], coefficient * l2normValues[n], columnValues[n]);
         status = PV_FAILURE;
      }
   }
   return PV_SUCCESS;
}
