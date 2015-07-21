function list = getResumeDownloadList(docRootNode)
    query = sprintf('/DownloadStatus/synsetTask[completed = 0]/wnid');
    result = XPathExecuteQuery(docRootNode, query);
    wnidList = cell(result.getLength());
    for i=0 : result.getLength()-1
        wnidList{(i +1)} = char(result.item(i).getFirstChild.getNodeValue);
    end
    list = wnidList;
end
