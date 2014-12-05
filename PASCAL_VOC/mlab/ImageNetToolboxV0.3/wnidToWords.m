function words = wnidToWords(structureXmlPath, wnid)
    if exist(structureXmlPath, 'file')
        %%docNode = xmlread(structureXmlPath); %% GTK: xmlread not implemented in octave
        docNode = xmlread(structureXmlPath);
    else
        error('%s does not contain structure_released.xml, please check your path', structureXmlPath);
    end
    
    query = sprintf('/ImageNetStructure//synset[@wnid="%s"]', wnid);
    result = XPathExecuteQuery(docNode, query);
    if result.getLength() == 0
        error('structure_released.xml does not contain %s, please check your wnid input\n', wnid);
    else
        words = strtrim(char(result.item(0).getAttribute('words')));
    end
end
