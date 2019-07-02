local PVModule = {}

--Global variable infinity declaration
infinity = math.huge
INFINITY = math.huge

local function appendString(table, str)
   assert(type(table)=="table")
   assert(type(str)=="string")
   table[#table+1] = str
end

--Prints a single parameter value to string for parameters in a group
local function valToString(paramStringTable, val)
   --NULL value
   if(type(val) == "function") then
      appendString(paramStringTable, "NULL")
   --Param sweep option
   elseif(type(val) == "table") then
      appendString(paramStringTable, "[")
      for i = 1, #val do
         valToString(paramStringTable, val[i])
         if(i ~= #val) then
            appendString(paramStringTable, ", ")
         end
      end
      appendString(paramStringTable, "]")
   --strings
   elseif(type(val) == "string") then
      appendString(paramStringTable, "\"")
      appendString(paramStringTable, val)
      appendString(paramStringTable, "\"")
   --boolean
   elseif(type(val) == "boolean") then
      if(val) then
         appendString(paramStringTable, "true")
      else
         appendString(paramStringTable, "false")
      end
   --infinity
   elseif(val == math.huge) then
      appendString(paramStringTable, "infinity")
   elseif(val == -math.huge) then
      appendString(paramStringTable, "-infinity")
   --numerical values
   else
      appendString(paramStringTable, tostring(val))
   end
end
      
-- Prints a key/value pair for a parameter within a group
local function printKeyValue(paramStringTable, key, val)
   appendString(paramStringTable, key)
   appendString(paramStringTable, " = ")
   valToString(paramStringTable, val)
   appendString(paramStringTable, ";\n")
end

--Prints an entire group, with groupType and Name specified
local function printGroup(paramStringTable, key, group)
   assert(group["groupType"] ~= nil)
   assert(type(group) == "table")

   if group["groupType"] == "BatchSweep" or group["groupType"] == "ParameterSweep" then

      appendString(paramStringTable, group["groupType"])
      appendString(paramStringTable, " ")
      appendString(paramStringTable, key)
      appendString(paramStringTable, " ")
      appendString(paramStringTable, " = {\n")

      for k,v in pairs(group) do
         if(k ~= "groupType") then
            appendString(paramStringTable, "    ");
            valToString(paramStringTable, v);
            appendString(paramStringTable, ";\n");
         end
      end

   else

      appendString(paramStringTable, group["groupType"])
      appendString(paramStringTable, " \"")
      appendString(paramStringTable, key)
      appendString(paramStringTable, "\" = {\n")
      for k,v in pairs(group) do
         if(k ~= "groupType") then
            appendString(paramStringTable, "    ")
            printKeyValue(paramStringTable, k, v)
         end
      end

   end
   appendString(paramStringTable, "};\n\n") --endOfGroup
end

--Converts parameterTable into a string suitable as a params file
function PVModule.createParamsFileString(parameterTable)
   paramStringTable = {}

   --First value to print out is always debugParams
   if(debugParsing == nil) then
      --Default to true
      printKeyValue(paramStringTable, "debugParsing", true)
   else
      printKeyValue(paramStringTable, "debugParsing", debugParsing)
   end

   --First group must be HyPerCol, TODO, change this
   for k,v in pairs(parameterTable) do
      if(type(v) == "table") then
         if(v["groupType"] == nil) then
            print("Error: group " .. k .. " does not have required parameter \"groupType\"")
            os.exit()
         end
         if(v["groupType"] == "HyPerCol") then
            printGroup(paramStringTable, k, v)
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
            printGroup(paramStringTable, k, v)
         end
      --TODO explict check for disable
      else
         --Not allowed, fatal error
         print("Group not table, fatal error")
         os.exit()
      end
   end
   paramsFile = table.concat(paramStringTable)
   return paramsFile
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

--Prints parameterTable table out to the console in a PV friendly way
function PVModule.printConsole(parameterTable)
   paramsFile = PVModule.createParamsFileString(parameterTable)
   io.write(paramsFile)
end


return PVModule
