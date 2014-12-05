function hashTable = buildClientHashtable(homefolder, key, value)    
    ht = java.util.Hashtable;

    docNode = xmlread(fullfile(homefolder, 'DownloadStatus.xml'));
    query = sprintf('/DownloadStatus/synsetTask/%s', key);
    keyResult = XPathExecuteQuery(docNode, query);

    query = sprintf('/DownloadStatus/synsetTask/%s', value);
    valueResult = XPathExecuteQuery(docNode, query);

    if keyResult.getLength() ~= valueResult.getLength()
        error('%s and %s is not match error\n', key, value);
    end

    for i = 0 : keyResult.getLength() - 1
        keyStr = char(keyResult.item(i).getFirstChild.getNodeValue);
        valueStr = char(valueResult.item(i).getFirstChild.getNodeValue);
        ht.put(keyStr, valueStr);
    end

    hashTable = ht;
end