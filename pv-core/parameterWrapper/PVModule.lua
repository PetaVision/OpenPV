local PVModule = {}

-- PVModule.utils = require "utils"

--Global variable infinity declaration
infinity = math.huge
INFINITY = math.huge

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
local function printGroup(key, group)
   assert(group["groupType"] ~= nil)
   assert(type(group) == "table")

   if group["groupType"] == "BatchSweep" or group["groupType"] == "ParameterSweep" then

      io.write(group["groupType"], " ", key, " ", " = {\n")

      for k,v in pairs(group) do
         if(k ~= "groupType") then
            io.write("    ");
            valToString(v
            );
            io.write(";\n");
         end
      end

   else

      io.write(group["groupType"], " \"", key, "\" = {\n")
      for k,v in pairs(group) do
         if(k ~= "groupType") then
            io.write("    ")
            printKeyValue(k, v)
         end
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

   --First group must be HyPerCol, TODO, change this
   for k,v in pairs(parameterTable) do
      if(type(v) == "table") then
         if(v["groupType"] == nil) then
            print("Error: group " .. k .. " does not have required parameter \"groupType\"")
            os.exit()
         end
         if(v["groupType"] == "HyPerCol") then
            printGroup(k, v)
            break
         end
      --TODO explict check for disable
      else
         --Not allowed, fatal error
         print("Group not table, fatal error")
         os.exit()
      end
   end

   --Iterate through rest of group
   for k,v in pairs(parameterTable) do
      --Exclude HyPerCols
      if(type(v) == "table") then
         assert(v["groupType"] ~= nil)
         if(v["groupType"] ~= "HyPerCol") then
            printGroup(k, v)
         end
      --TODO explict check for disable
      else
         --Not allowed, fatal error
         print("Group not table, fatal error")
         os.exit()
      end
   end
end

--Adds a collection of groups by copying.
function PVModule.addMultiGroups(parameterTable, group)
   assert(type(parameterTable) == "table")
   assert(type(group) == "table")
   for k,v in pairs(group) do
      assert(type(v) == "table")
      --This function will make a copy, not reference
      PVModule.addGroup(parameterTable, k, v)
   end
end


--Adds a group by copying. Note that group can be an existing group specified
--If overwrites is not nil, we will append overwrites table to parameterTable, clobbering
--previous parameterTable values
--Can specify an array of groups or an individual group
function PVModule.addGroup(parameterTable, newKey, group, overwrites)
   --Sanity assertions
   assert(type(parameterTable) == "table")
   assert(type(group) == "table")

   --Check that newKey does not exist in parameterTable
   if(parameterTable[newKey] ~= nil) then
      print("Error: Group " .. newKey .. " already exists in the parameterTable")
      os.exit()
   end

   --Make a copy of group as opposed to referencing the same table group to newGroup
   newGroup = PVModule.deepCopy(group)
   --overwrites is optional - nil if not specified
   if(overwrites ~= nil) then
      assert(type(overwrites) == "table")
      --Overwrite parameters in group
      for k,v in pairs(overwrites) do
         if(newGroup[k] == nil) then
            print("Overwrite error: parameter ".. k .." does not exist in " .. group["groupType"] .. " \"".. newKey.. "\"")
            os.exit()
         end
         --deepCopy may not be nessessary here, as overwrites is usually a user defined group
         newGroup[k] = PVModule.deepCopy(v);
      end
   end
   --Length of list + 1
   parameterTable[newKey] = newGroup;
end

--Deprecated in favor of accessing by parameterTable["name"]

----Returns the REFERENCE of the object with the groupName of "name" from the parameterTable
--function PVModule.getGroupFromName(parameterTable, name)
--   assert(type(parameterTable) == "table")
--   assert(type(name) == "string")
--   for k,v in pairs(parameterTable) do
--      --Here, parameterTable is a list, k is an integer
--      assert(type(v) == "table")
--      assert(v["groupName"])
--      if(v["groupName"] == name) then
--         return parameterTable[k]
--      end
--   end
--   return nil
--end

--Function make a deep copy of obj and return new object
--Call function with only 1 parameter
--newTable = PVModule.deepCopy(sourceTable)
function PVModule.deepCopy(obj, seen)
   -- Handle non-tables and previously-seen tables.
   if type(obj) ~= 'table' then return obj end
   if seen and seen[obj] then
      io.write("deepCopy ran into a recursive list, fatal error\n")
      os.exit()
   end
   -- New table; mark it as seen and copy recursively.
   local s = seen or {}
   --Save metatables (op overloading tables)
   local res = setmetatable({}, getmetatable(obj))
   s[obj] = res
   --Recursive copy
   for k, v in pairs(obj) do res[PVModule.deepCopy(k, s)] = PVModule.deepCopy(v, s) end
   return res
end

function PVModule.batchSweep(pvParams, group, param, values)
   assert(group ~= nil)
   assert(param ~= nil)
   key = string.format('"%s":%s', group, param);
   local sweep = {};
   for i,v in ipairs(values) do
      sweep[i] = v;
   end
   sweep['groupType'] = "BatchSweep";
   PVModule.addGroup(pvParams, key, sweep);
end

function PVModule.paramSweep(pvParams, group, param, values)
   assert(group ~= nil)
   assert(param ~= nil)
   key = string.format('"%s":%s', group, param);
   local sweep = {};
   for i,v in ipairs(values) do
      sweep[i] = v;
   end
   sweep['groupType'] = "ParameterSweep";
   PVModule.addGroup(pvParams, key, sweep);
end



return PVModule
