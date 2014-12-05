function hashtable = serverReleaseStatusToHash(homefolder)
    ht = java.util.Hashtable;
    docNode = xmlread(fullfile(homefolder, 'ReleaseStatus.xml'));
    query = sprintf('/ReleaseStatus/images/synsetInfos/synset');
    result = XPathExecuteQuery(docNode, query);
    for i = 0 : result.getLength() - 1
        t = result.item(i);
        offset = char(t.getAttribute('wnid'));
        version = char(t.getAttribute('version'));
        ht.put(offset, version);
    end
    hashtable = ht;
end
