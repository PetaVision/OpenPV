/*
 * CompareParamsFiles.cpp
 *
 *  Created on: Dec 17, 2018
 *      Author: pschultz
 */

#include "CompareParamsFiles.hpp"
#include "include/pv_common.h"

namespace PV {

int compareParamsFiles(
      std::string const &paramsFile1,
      std::string const &paramsFile2,
      MPI_Comm mpiComm) {
   int status = PV_SUCCESS;
   PVParams params1{paramsFile1.c_str(), INITIAL_LAYER_ARRAY_SIZE, mpiComm};
   PVParams params2{paramsFile2.c_str(), INITIAL_LAYER_ARRAY_SIZE, mpiComm};

   // create a map between groups in paramsFile1 and those in paramsFile2.
   std::map<ParameterGroup *, ParameterGroup *> parameterGroupMap;
   char const *groupName = nullptr;
   for (int idx = 0; (groupName = params1.groupNameFromIndex(idx)) != nullptr; idx++) {
      ParameterGroup *g1 = params1.group(groupName);
      ParameterGroup *g2 = params2.group(groupName);
      if (g2 == nullptr) {
         ErrorLog().printf(
               "Group name \"%s\" is in \"%s\" but not in \"%s\".\n",
               groupName,
               paramsFile1.c_str(),
               paramsFile2.c_str());
         status = PV_FAILURE;
      }
      else {
         parameterGroupMap.emplace(std::make_pair(g1, g2));
      }
   }
   for (int idx = 0; (groupName = params2.groupNameFromIndex(idx)) != nullptr; idx++) {
      if (params1.group(groupName) == nullptr) {
         ErrorLog().printf(
               "Group name \"%s\" is in \"%s\" but not in \"%s\".\n",
               groupName,
               paramsFile2.c_str(),
               paramsFile1.c_str());
         status = PV_FAILURE;
      }
   }

   for (auto &p : parameterGroupMap) {
      status |= compareParameterGroups(p.first, p.second);
   }
   return status;
}

int compareParameterGroups(ParameterGroup *group1, ParameterGroup *group2) {
   int status = PV_SUCCESS;
   if (strcmp(group1->name(), group2->name())) {
      ErrorLog().printf(
            "Groups have different names (\"%s\" versus \"%s\")\n", group1->name(), group2->name());
      status = PV_FAILURE;
   }
   if (strcmp(group1->getGroupKeyword(), group2->getGroupKeyword())) {
      ErrorLog().printf(
            "Keywords for group \"%s\" do not match (\"%s\" versus \"%s\").\n",
            group1->name(),
            group1->getGroupKeyword(),
            group2->getGroupKeyword());
      status = PV_FAILURE;
   }
   ParameterStack *numericStack1 = group1->copyStack();
   ParameterStack *numericStack2 = group2->copyStack();
   status |= compareParameterNumericStacks(group1->name(), numericStack1, numericStack2);
   delete numericStack1;
   delete numericStack2;

   ParameterArrayStack *arrayStack1 = group1->copyArrayStack();
   ParameterArrayStack *arrayStack2 = group2->copyArrayStack();
   status |= compareParameterArrayStacks(group1->name(), arrayStack1, arrayStack2);
   delete arrayStack1;
   delete arrayStack2;

   ParameterStringStack *stringStack1 = group1->copyStringStack();
   ParameterStringStack *stringStack2 = group2->copyStringStack();
   status |= compareParameterStringStacks(group1->name(), stringStack1, stringStack2);
   delete stringStack1;
   delete stringStack2;

   return status;
}

int compareParameterNumericStacks(
      char const *groupName,
      ParameterStack *stack1,
      ParameterStack *stack2) {
   int status = PV_SUCCESS;

   // Vector of booleans indicating for each element of stack2 whether the name exists in stack1
   std::vector<bool> inStack1(stack2->size(), false);
   for (int i = 0; i < stack1->size(); i++) {
      Parameter *param1      = stack1->peek(i);
      char const *paramName1 = param1->name();
      bool found             = false;
      for (int j = 0; j < stack2->size(); j++) {
         Parameter *param2      = stack2->peek(j);
         char const *paramName2 = param2->name();
         if (!strcmp(paramName1, paramName2)) {
            found       = true;
            inStack1[j] = true;
            if (param1->value() != param2->value()) {
               ErrorLog().printf(
                     "Parameter \"%s\" in group \"%s\" has different values (%f versus %f).\n",
                     paramName1,
                     groupName,
                     param1->value(),
                     param2->value());
               status = PV_FAILURE;
            }
            break;
         }
      }
      if (!found) {
         ErrorLog().printf(
               "Parameter \"%s\" was found in group \"%s\" of one stack but not the other.\n",
               paramName1,
               groupName);
         status = PV_FAILURE;
      }
   }
   for (int j = 0; j < stack2->size(); j++) {
      if (!inStack1[j]) {
         ErrorLog().printf(
               "Parameter \"%s\" was found in group \"%s\" of one stack but not the other.\n",
               stack2->peek(j)->name(),
               groupName);
         status = PV_FAILURE;
      }
   }
   return status;
}

int compareParameterArrayStacks(
      char const *groupName,
      ParameterArrayStack *stack1,
      ParameterArrayStack *stack2) {
   int status = PV_SUCCESS;

   // Vector of booleans indicating for each element of stack2 whether the name exists in stack1
   std::vector<bool> inStack1(stack2->size(), false);
   for (int i = 0; i < stack1->size(); i++) {
      ParameterArray *param1 = stack1->peek(i);
      char const *paramName1 = param1->name();
      bool found             = false;
      for (int j = 0; j < stack2->size(); j++) {
         ParameterArray *param2 = stack2->peek(j);
         char const *paramName2 = param2->name();
         if (!strcmp(paramName1, paramName2)) {
            found       = true;
            inStack1[j] = true;
            status |= compareParameterArray(groupName, param1, param2);
            break;
         }
      }
      if (!found) {
         ErrorLog().printf(
               "Array parameter \"%s\" was found in group \"%s\" of one stack but not the other.\n",
               paramName1,
               groupName);
         status = PV_FAILURE;
      }
   }
   for (int j = 0; j < stack2->size(); j++) {
      if (!inStack1[j]) {
         ErrorLog().printf(
               "Array parameter \"%s\" was found in group \"%s\" of one stack but not the other.\n",
               stack2->peek(j)->name(),
               groupName);
         status = PV_FAILURE;
      }
   }
   return status;
}

int compareParameterArray(char const *groupName, ParameterArray *array1, ParameterArray *array2) {
   int status = PV_SUCCESS;
   if (strcmp(array1->name(), array2->name())) {
      ErrorLog().printf(
            "Arrays have different names (\"%s\" versus \"%s\")\n", array1->name(), array2->name());
   }
   if (array1->getArraySize() != array2->getArraySize()) {
      ErrorLog().printf(
            "Array \"%s\" in group \"%s\" differs in size (%d versus %d).\n",
            array1->name(),
            groupName,
            array1->getArraySize(),
            array2->getArraySize());
   }
   int size1, size2;
   double const *values1 = array1->getValuesDbl(&size1);
   double const *values2 = array2->getValuesDbl(&size2);
   int size              = size1 <= size2 ? size1 : size2;
   int badIndex          = 0;
   for (int i = 0; i < size; i++) {
      if (values1[i] != values2[i]) {
         badIndex = i + 1;
         ErrorLog().printf(
               "Group %s, array %s, index %d differs (%f versus %f).\n",
               groupName,
               array1->name(),
               badIndex,
               values1[i],
               values2[i]);
         status = PV_FAILURE;
      }
   }
   return status;
}

int compareParameterStringStacks(
      char const *groupName,
      ParameterStringStack *stack1,
      ParameterStringStack *stack2) {
   int status = PV_SUCCESS;

   // Vector of booleans indicating for each element of stack2 whether the name exists in stack1
   std::vector<bool> inStack1(stack2->size(), false);
   for (int i = 0; i < stack1->size(); i++) {
      ParameterString *param1 = stack1->peek(i);
      char const *paramName1  = param1->getName();
      bool found              = false;
      for (int j = 0; j < stack2->size(); j++) {
         ParameterString *param2 = stack2->peek(j);
         char const *paramName2  = param2->getName();
         if (!strcmp(paramName1, paramName2)) {
            found       = true;
            inStack1[j] = true;
            status |= compareParameterString(groupName, param1, param2);
            break;
         }
      }
      if (!found) {
         ErrorLog().printf(
               "String parameter \"%s\" was found in group \"%s\" of one stack but not the "
               "other.\n",
               paramName1,
               groupName);
         status = PV_FAILURE;
      }
   }
   for (int j = 0; j < stack2->size(); j++) {
      if (!inStack1[j]) {
         ErrorLog().printf(
               "String parameter \"%s\" was found in group \"%s\" of one stack "
               "but not the other.\n",
               stack2->peek(j)->getName(),
               groupName);
         status = PV_FAILURE;
      }
   }
   return status;
}

int compareParameterString(
      char const *groupName,
      ParameterString *string1,
      ParameterString *string2) {
   int status = PV_SUCCESS;
   if (strcmp(string1->getName(), string2->getName())) {
      ErrorLog().printf(
            "ParameterStrings have different names (\"%s\" versus \"%s\")\n",
            string1->getName(),
            string2->getName());
      status = PV_FAILURE;
   }
   char const *nullString = "(null)";
   char const *value1     = string1->getValue();
   if (value1 == nullptr) {
      value1 = nullString;
   }
   char const *value2 = string2->getValue();
   if (value2 == nullptr) {
      value2 = nullString;
   }
   if (strcmp(value1, value2)) {
      ErrorLog().printf(
            "String parameter \"%s\" in group \"%s\" differs (\"%s\" versus \"%s\").\n",
            string1->getName(),
            groupName,
            value1,
            value2);
      status = PV_FAILURE;
   }
   return status;
}

} // end namespace PV
