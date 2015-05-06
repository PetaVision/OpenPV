local PVModule = {}

local utils = require "utils"

--local PVModule.Validated = false

--Prints a single parameter value to string for parameters in a group
local function valToString(val)
   --NULL value
   if(val == nil) then
      io.write("NULL")
   --Param sweep option
   elseif(type(val) == "table") then
      io.write("[")
      for i = 1, #val do
         valToString(val[i])
         if(i ~= #val) then
            io.write(", ")
         end
      end
      io.write("]")
   --strings
   elseif(type(val) == "string") then
      io.write("\"", val, "\"")
   --boolean
   elseif(type(val) == "boolean") then
      if(val) then
         io.write("true")
      else
         io.write("false")
      end
   --infinity
   elseif(val == math.huge) then
      io.write("infinity")
   elseif(val == -math.huge) then
      io.write("-infinity")
   --numerical values
   else
      io.write(val)
   end
end
      
-- Prints a key/value pair for a parameter within a group
local function printKeyValue(key, val)
   io.write(key ," = ")
   valToString(val)
   io.write(";\n")
end

--Prints an entire group, with groupType and Name specified
local function printGroup(group)
   assert(group["groupType"] ~= nil)
   assert(group["groupName"] ~= nil)
   assert(type(group) == "table")

   io.write(group["groupType"], " \"", group["groupName"], "\" = {\n")

   for k,v in pairs(group) do
      if(k ~= "groupType" and k ~= "groupName") then
         io.write("   ")
         printKeyValue(k, v)
      end
   end
   io.write("};\n\n") --endOfGroup
end


--local function validateVal(group)
--
--end
--
--local function validateGroup(group)
--
--end
--
----Validates syntax of parameterTable
--function PVModule.validate(parameterTable)
--   --Parameter table must be a continuous array, 1 to numGroups
--   if(~isArray(group)) then
--      print("Group specified not an array")
--      os.exit()
--   end
--end

--Prints parameterTable table out to the console in a PV friendly way
function PVModule.printConsole(parameterTable)
   --Validate before print
   --PVModule.validate(parameterTable)
   
   --First value to print out is always debugParams
   if(debugParsing == nil) then
      --Default to true
      printKeyValue("debugParsing", true)
   else
      printKeyValue("debugParsing", debugParsing)
   end
   --Iterate through rest of group
   for k,v in pairs(parameterTable) do
      if(type(v) == "table") then
         printGroup(v)
      --TODO explict check for disable
      else
         --Not allowed, fatal error
         print("Group not table, fatal error")
         os.exit()
      end
   end
end

--Adds a group by copying. Note that group can be an existing group specified
--If overwrites is not nil, we will append overwrites table to parameterTable, clobbering
--previous parameterTable values
--Can specify an array of groups or an individual group
function PVModule.addGroup(parameterTable, group, overwrites)
   --Sanity assertions
   assert(type(parameterTable) == "table")
   assert(parameterTable[#parameterTable+1] == nil)
   assert(type(group) == "table")
   --If it's an array, it will be an array of groups
   if(utils.isArray(group)) then
      for k,v in pairs(group) do
         assert(type(v) == "table")
         --This function will make a copy, not reference
         PVModule.addGroup(parameterTable, v)
      end
   else
      --Make a copy of group as opposed to referencing the same table group to newGroup
      newGroup = utils.deepCopy(group)
      --overwrites is optional - nil if not specified
      if(overwrites ~= nil) then
         assert(type(overwrites) == "table")
         --Overwrite parameters in group
         for k,v in pairs(overwrites) do
            if(newGroup[k] == nil) then
               io.write("Overwrite error: parameter ", k, " does not exist in ", group["groupType"], " \"", group["groupName"], "\"\n")
               os.exit()
            end
            --deepCopy may not be nessessary here
            newGroup[k] = utils.deepCopy(v);
         end
      end
      --Length of list + 1
      parameterTable[#parameterTable+1] = newGroup;
   end
end

--Returns the REFERENCE of the object with the groupName of "name" from the parameterTable
function PVModule.getGroupFromName(parameterTable, name)
   assert(type(parameterTable) == "table")
   assert(type(name) == "string")
   for k,v in pairs(parameterTable) do
      --Here, parameterTable is a list, k is an integer
      assert(type(v) == "table")
      assert(v["groupName"])
      if(v["groupName"] == name) then
         return parameterTable[k]
      end
   end
   return nil
end

return PVModule
