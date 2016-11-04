/*
 * DryRunFlagTest.cpp
 *
 */

#include "AlwaysFailsLayer.hpp"
#include <columns/buildandrun.hpp>
#include <sys/types.h>
#include <unistd.h>
#include <utils/PVAssert.hpp>

int deleteOutputDirectory(PV::Communicator *comm);
void compareParamsFiles(char const *paramsFile1, char const *paramsFile2, PV::Communicator *comm);
void compareParameterGroups(PV::ParameterGroup *group1, PV::ParameterGroup *group2);
void compareParameterNumericStacks(
      char const *groupName,
      PV::ParameterStack *stack1,
      PV::ParameterStack *stack2);
void compareParameterNumeric(
      char const *groupName,
      PV::Parameter *parameter1,
      PV::Parameter *parameter2);
void compareParameterArrayStacks(
      char const *groupName,
      PV::ParameterArrayStack *stack1,
      PV::ParameterArrayStack *stack2);
void compareParameterArray(
      char const *groupName,
      PV::ParameterArray *array1,
      PV::ParameterArray *array2);
void compareParameterStringStacks(
      char const *groupName,
      PV::ParameterStringStack *stack1,
      PV::ParameterStringStack *stack2);
void compareParameterString(
      char const *groupName,
      PV::ParameterString *string1,
      PV::ParameterString *string2);

int main(int argc, char *argv[]) {

   int status = PV_SUCCESS;

   PV::PV_Init pv_obj(&argc, &argv, false /*allowUnrecognizedArguments*/);
   pv_obj.registerKeyword("AlwaysFailsLayer", Factory::create<AlwaysFailsLayer>);

   pv_obj.setDryRunFlag(true);

   if (pv_obj.isExtraProc()) {
      return EXIT_SUCCESS;
   }

   FatalIf(
         pv_obj.getParamsFile() != nullptr,
         "%s should be called without the -p argument; the necessary params file is hard-coded.\n");
   pv_obj.setParams("input/DryRunFlagTest.params");

   int rank = pv_obj.getCommunicator()->globalCommRank();

   status = deleteOutputDirectory(pv_obj.getCommunicator());
   if (status != PV_SUCCESS) {
      Fatal().printf("%s: error cleaning generated files from any previous run.\n", argv[0]);
   }

   status = buildandrun(&pv_obj);

   if (status != PV_SUCCESS) {
      Fatal().printf("%s: running with dry-run flag set failed on process %d.\n", argv[0], rank);
   }

   compareParamsFiles("output/pv.params", "input/correct.params", pv_obj.getCommunicator());
}

int deleteOutputDirectory(PV::Communicator *comm) {
   int status = PV_SUCCESS;
   if (comm->globalCommRank() == 0) {
      if (system("rm -rf output") != PV_SUCCESS) {
         status = PV_FAILURE;
      }
   }
   MPI_Bcast(&status, 1, MPI_INT, 0, comm->communicator());
   return status;
}

void compareParamsFiles(char const *paramsFile1, char const *paramsFile2, PV::Communicator *comm) {
   PV::PVParams params1{paramsFile1, INITIAL_LAYER_ARRAY_SIZE, comm};
   PV::PVParams params2{paramsFile2, INITIAL_LAYER_ARRAY_SIZE, comm};

   // create a map between groups in paramsFile1 and those in paramsFile2.
   std::map<PV::ParameterGroup *, PV::ParameterGroup *> parameterGroupMap;
   char const *groupName = nullptr;
   for (int idx = 0; (groupName = params1.groupNameFromIndex(idx)) != nullptr; idx++) {
      PV::ParameterGroup *g1 = params1.group(groupName);
      PV::ParameterGroup *g2 = params2.group(groupName);
      FatalIf(
            g2 == nullptr,
            "Group name \"%s\" is in \"%s\" but not in \"%s\".\n",
            groupName,
            paramsFile1,
            paramsFile2);
      parameterGroupMap.emplace(std::make_pair(g1, g2));
   }
   for (int idx = 0; (groupName = params2.groupNameFromIndex(idx)) != nullptr; idx++) {
      FatalIf(
            params1.group(groupName) == nullptr,
            "Group name \"%s\" is in \"%s\" but not in \"%s\".\n",
            groupName,
            paramsFile2,
            paramsFile1);
   }

   for (auto &p : parameterGroupMap) {
      compareParameterGroups(p.first, p.second);
   }
   return;
}

void compareParameterGroups(PV::ParameterGroup *group1, PV::ParameterGroup *group2) {
   pvAssert(!strcmp(group1->name(), group2->name()));
   FatalIf(
         strcmp(group1->getGroupKeyword(), group2->getGroupKeyword()),
         "Keywords for group \"%s\" do not match (\"%s\" versus \"%s\").\n",
         group1->name(),
         group1->getGroupKeyword(),
         group2->getGroupKeyword());
   PV::ParameterStack *numericStack1 = group1->copyStack();
   PV::ParameterStack *numericStack2 = group2->copyStack();
   compareParameterNumericStacks(group1->name(), numericStack1, numericStack2);

   PV::ParameterArrayStack *arrayStack1 = group1->copyArrayStack();
   PV::ParameterArrayStack *arrayStack2 = group2->copyArrayStack();
   compareParameterArrayStacks(group1->name(), arrayStack1, arrayStack2);

   PV::ParameterStringStack *stringStack1 = group1->copyStringStack();
   PV::ParameterStringStack *stringStack2 = group2->copyStringStack();
   compareParameterStringStacks(group1->name(), stringStack1, stringStack2);
}

void compareParameterNumericStacks(
      char const *groupName,
      PV::ParameterStack *stack1,
      PV::ParameterStack *stack2) {
   FatalIf(
         stack1->size() != stack2->size(),
         "Numeric stacks for \"%s\" have different sizes: %d versus %d.\n",
         groupName,
         stack1->size(),
         stack2->size());
   // create a map between stack1 and stack2
   int const size = stack1->size();
   std::map<PV::Parameter *, PV::Parameter *> parameterMap;
   for (int i = 0; i < size; i++) {
      PV::Parameter *param1  = stack1->peek(i);
      char const *paramName1 = param1->name();
      bool found             = false;
      for (int j = 0; j < size; j++) {
         PV::Parameter *param2  = stack2->peek(j);
         char const *paramName2 = param2->name();
         if (!strcmp(paramName1, paramName2)) {
            parameterMap.emplace(std::make_pair(param1, param2));
            found = true;
            break;
         }
      }
      if (!found) {
         Fatal() << "Parameter \"" << paramName1 << "\" was found in group \"" << groupName
                 << "\" of one stack but not the other.\n";
      }
   }
   pvAssert(parameterMap.size() == size);

   for (auto &p : parameterMap) {
      compareParameterNumeric(groupName, p.first, p.second);
   }
}

void compareParameterNumeric(
      char const *groupName,
      PV::Parameter *parameter1,
      PV::Parameter *parameter2) {
   pvAssert(!strcmp(parameter1->name(), parameter2->name()));
   FatalIf(
         parameter1->value() != parameter2->value(),
         "Numeric parameter \"%s\" in group \"%s\" differs (%f versus %f).\n",
         parameter1->name(),
         groupName,
         parameter1->value(),
         parameter2->value());
}

void compareParameterArrayStacks(
      char const *groupName,
      PV::ParameterArrayStack *stack1,
      PV::ParameterArrayStack *stack2) {
   FatalIf(
         stack1->size() != stack2->size(),
         "Numeric stacks for \"%s\" have different sizes: %d versus %d.\n",
         groupName,
         stack1->size(),
         stack2->size());
   // create a map between stack1 and stack2
   int const size = stack1->size();
   std::map<PV::ParameterArray *, PV::ParameterArray *> parameterArrayMap;
   for (int i = 0; i < size; i++) {
      PV::ParameterArray *param1 = stack1->peek(i);
      char const *paramName1     = param1->name();
      bool found                 = false;
      for (int j = 0; j < size; j++) {
         PV::ParameterArray *param2 = stack2->peek(j);
         char const *paramName2     = param2->name();
         if (!strcmp(paramName1, paramName2)) {
            parameterArrayMap.emplace(std::make_pair(param1, param2));
            found = true;
            break;
         }
      }
      if (!found) {
         Fatal() << "Parameter \"" << paramName1 << "\" was found in group \"" << groupName
                 << "\" of one stack but not the other.\n";
      }
   }
   pvAssert(parameterArrayMap.size() == size);

   for (auto &p : parameterArrayMap) {
      compareParameterArray(groupName, p.first, p.second);
   }
}

void compareParameterArray(
      char const *groupName,
      PV::ParameterArray *array1,
      PV::ParameterArray *array2) {
   pvAssert(!strcmp(array1->name(), array2->name()));
   FatalIf(
         array1->getArraySize() != array2->getArraySize(),
         "Array \"%s\" in group \"%s\" differs in size (%d versus %d).\n",
         array1->name(),
         groupName,
         array1->getArraySize(),
         array2->getArraySize());
   int size              = array1->getArraySize();
   double const *values1 = array1->getValuesDbl(&size);
   double const *values2 = array2->getValuesDbl(&size);
   int badIndex          = 0;
   for (int i = 0; i < size; i++) {
      if (values1[i] != values2[i]) {
         badIndex = i + 1;
         ErrorLog() << "Group " << groupName << ", array " << array1->name() << ", index "
                    << badIndex << " differs (" << values1[i] << " versus " << values2[i] << ").\n";
      }
   }
   if (badIndex > 0) {
      exit(EXIT_FAILURE);
   }
}

void compareParameterStringStacks(
      char const *groupName,
      PV::ParameterStringStack *stack1,
      PV::ParameterStringStack *stack2) {
   FatalIf(
         stack1->size() != stack2->size(),
         "Numeric stacks for \"%s\" have different sizes: %d versus %d.\n",
         groupName,
         stack1->size(),
         stack2->size());
   // create a map between stack1 and stack2
   int const size = stack1->size();
   std::map<PV::ParameterString *, PV::ParameterString *> parameterStringMap;
   for (int i = 0; i < size; i++) {
      PV::ParameterString *param1 = stack1->peek(i);
      char const *paramName1      = param1->getName();
      bool found                  = false;
      for (int j = 0; j < size; j++) {
         PV::ParameterString *param2 = stack2->peek(j);
         char const *paramName2      = param2->getName();
         if (!strcmp(paramName1, paramName2)) {
            parameterStringMap.emplace(std::make_pair(param1, param2));
            found = true;
            break;
         }
      }
      if (!found) {
         Fatal() << "Parameter \"" << paramName1 << "\" was found in group \"" << groupName
                 << "\" of one stack but not the other.\n";
      }
   }
   pvAssert(parameterStringMap.size() == size);

   for (auto &p : parameterStringMap) {
      compareParameterString(groupName, p.first, p.second);
   }
}

void compareParameterString(
      char const *groupName,
      PV::ParameterString *string1,
      PV::ParameterString *string2) {
   pvAssert(!strcmp(string1->getName(), string2->getName()));
   char const *nullString = "(null)";
   char const *value1     = string1->getValue();
   if (value1 == nullptr) {
      value1 = nullString;
   }
   char const *value2 = string2->getValue();
   if (value2 == nullptr) {
      value2 = nullString;
   }
   FatalIf(
         strcmp(value1, value2),
         "String parameter \"%s\" in group \"%s\" differs (\"%s\" versus \"%s\").\n",
         string1->getName(),
         groupName,
         value1,
         value2);
}
