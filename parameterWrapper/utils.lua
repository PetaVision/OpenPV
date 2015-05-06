local utils = {}

--Global variable infinity declaration
INFINITY = math.huge

--Function make a deep copy of obj and return new object
--Call function with only 1 parameter
--newTable = utils.deepCopy(sourceTable)
function utils.deepCopy(obj, seen)
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
   for k, v in pairs(obj) do res[utils.deepCopy(k, s)] = utils.deepCopy(v, s) end
   return res
end

function utils.isArray(obj)
   local i = 0
   --Lua always starts from obj[1] and iterates continously
   for _ in pairs(obj) do
      i = i + 1
      --Therefore, any entry must not be nil
      if obj[i] == nil then return false end
   end
   return true
end

return utils
