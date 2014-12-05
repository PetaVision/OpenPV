function  reply = checkResumeNeeded(docNode)
    reply = 'N';
    query = sprintf('/DownloadStatus/synsetTask[completed = 0]');
    result = XPathExecuteQuery(docNode, query);
    if result.getLength() > 0
        reply = input('The last download task is not completed.\nPress Y to resume the unfinished download task and then download the new task\npress N to start the new task only: ', 's');
        if isempty(reply)
            reply = 'Y';
        end
    end
    reply = upper(reply);
end
