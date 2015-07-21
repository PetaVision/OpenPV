function gloss = wnidToGloss(structureXmlPath, wnid)
    if exist(structureXmlPath, 'file')
        docNode = xmlread(structureXmlPath);
    else
        error('%s does not contain structure_released.xml, please check your path\n', structureXmlPath);
    end
    query = sprintf('/ImageNetStructure//synset[@wnid="%s"]', wnid);
    result = XPathExecuteQuery(docNode, query);
    if result.getLength == 0
        error('structure_released.xml does not contain %s, please check your wnid input\n', wnid);
    else
        gloss =  strtrim(char(result.item(0).getAttribute('gloss')));
    end
end
