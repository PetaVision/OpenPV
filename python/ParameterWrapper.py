from copy import deepcopy

infinity = float('inf')
INFINITY = float('inf')
nil      = None
NULL     = None
null     = None
true     = True
false    = False

# The param file is stored as a list of strings to prevent memory issues
# with large strings.
def appendString(stringList, string):
    assert(type(stringList) == type([]))
    assert(type(string) == type(''))
    stringList.append(string)

def appendValue(stringList, val):
    if val == None:
        appendString(stringList, 'NULL')
    elif type(val) == type([]):
        appendString(stringList, str(val))
    elif type(val) == type(''):
        appendString(stringList, '\"' + val + '\"')
    elif type(val) == type(True):
        appendString(stringList, str(val).lower())
    elif val == float('inf') or val == -float('inf'):
        appendString(stringList, str(val) + 'inity')
    else:
        appendString(stringList, str(val))

def appendKeyValue(stringList, key, val):
    appendString(stringList, '    ' + key.ljust(30) + '= ')
    appendValue(stringList, val)
    appendString(stringList, ';')

def appendGroup(stringList, name, group):
    assert(type(group) == type({}))
    assert('groupType' in group)
    if group['groupType'] == 'BatchSweep' or group['groupType'] == "ParameterSweep":
        #TODO
        print('BatchSweep and ParameterSweep not implemented')
        return
    else:
        appendString(stringList, group['groupType'] + ' \"' + name + '\" = {\n')
        for key, value in group.items():
            if key == 'groupType':
                continue
            appendKeyValue(stringList, key, value)
            appendString(stringList, '\n')
    appendString(stringList, '};\n\n')


def createParamsFileString(parameterTable, debugParsing=True):
    stringList = []
    
    appendKeyValue(stringList, 'debugParsing', debugParsing)
    appendString(stringList, '\n\n')

    # prints HyPerCol first, is this necessary?
    for key, value in parameterTable.items():
        if type(value) == type({}):
            if value['groupType'] == None:
                print('Error: group ' + key + ' does not have required parameter \"groupType\"')
                return
            if value['groupType'] == 'HyPerCol':
                appendGroup(stringList, key, value)
                break
        else:
            print('Error: group is not a dictionary')
            return
                
    for key, value in parameterTable.items():
        if type(value) == type({}):
            if value['groupType'] == 'HyPerCol':
                continue
            appendGroup(stringList, key, value)
        else:
            print('Error: group is not a dictionary')
            return

    return ''.join(stringList) 


def addGroup(parameterTable, newKey, group, overwrites=None):
    assert(type(parameterTable) == type({}))
    assert(type(group) == type({}))

    if newKey in parameterTable:
        print('Error: Group ' + newKey + ' already exists')
        return

    newGroup = deepcopy(group)
    
    if overwrites != None:
        assert(type(overwrites) == type({}))
        for key, value in overwrites.items():
            if key not in newGroup:
                print('Error: parameter ' + key + ' does not exist in ' + newKey)
                return
            newGroup[key] = deepcopy(value)

    parameterTable[newKey] = newGroup

def addMultiGroups(parameterTable, groups):
    assert(type(parameterTable) == type({}))
    assert(type(groups) == type({}))
    for key, value in groups.items():
        assert(type(value) == type({}))
        addGroup(parameterTable, key, value)


def batchSweep(parameterTable, group, param, values):
    #TODO
    print('BatchSweep not implemented')
    return

def paramSweep(parameterTable, group, param, values):
    #TODO
    print('ParameterSweep not implemented')
    return


