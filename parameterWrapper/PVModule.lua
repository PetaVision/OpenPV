local PVModule = {}

--Global variable infinity declaration
INFINITY = math.huge

local function valToString(val)
   --NULL value
   if(val == nil) then
      --TODO double check this, I think lowercase null
      write("null")
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
   --strings
   elseif(val == "string") then
      io.write("\"", val, "\"")
   else
      io.write(val)
   end
end
      
local function printKeyValue(key, val)
   io.write(key ," = ")
   valToString(val)
   io.write(";\n")
end

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

local function deepCopy(obj, seen)
   -- Handle non-tables and previously-seen tables.
   if type(obj) ~= 'table' then return obj end
   if seen and seen[obj] then
      io.write("deepCopy ran into a recursive list, fatal error\n")
      os.exit()
   end

   -- New table; mark it as seen an copy recursively.
   local s = seen or {}
   local res = setmetatable({}, getmetatable(obj))
   s[obj] = res
   for k, v in pairs(obj) do res[deepCopy(k, s)] = deepCopy(v, s) end
   return res
end

function PVModule.printConsole(parameterTable)
   --First value to print out is always debugParams
   --debugParsing should be a global variable, TODO, change this?
   printKeyValue("debugParsing", debugParsing)
   --Iterate through rest of group
   for k,v in pairs(parameterTable) do
      --Skip debugParams, already been done
      if(k ~= "debugParsing") then
         if(type(v) == "table") then
            printGroup(v)
         else
            printKeyValue(k, v)
         end
      end
   end
end

--Adds a group by copying. Note that group can be an existing group specified
--If overwrites is not nil, we will append overwrites table to parameterTable, clobbering
--previous parameterTable values
function PVModule.addGroup(parameterTable, group, overwrites)
   --Sanity assertions
   assert(type(parameterTable) == "table")
   assert(parameterTable[#parameterTable+1] == nil)
   assert(type(group) == "table")
   --Make a copy of group as opposed to referencing the same table group to newGroup
   newGroup = deepCopy(group)
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
         newGroup[k] = deepCopy(v);
      end
   end
   --Length of list + 1
   parameterTable[#parameterTable+1] = deepCopy(newGroup);
end

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
