function message = verifyUserAccesskeyPair(username, accesskey)
    verificationURL = sprintf('http://www.image-net.org/api/text/verification.user_accesskey.php?username=%s&accesskey=%s', username, accesskey);
    url = java.net.URL(verificationURL);
    is = openStream(url);
    isr = java.io.InputStreamReader(is);
    br = java.io.BufferedReader(isr);
    message = readLine(br);
end
