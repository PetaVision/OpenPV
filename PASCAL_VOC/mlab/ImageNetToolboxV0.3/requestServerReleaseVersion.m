function serverReleaseVersion = requestServerReleaseVersion()
    serverVersionURL = 'http://www.image-net.org/api/text/imagenet.check_latest_version.php';
    url = java.net.URL(serverVersionURL);
    is = openStream(url);
    isr = java.io.InputStreamReader(is);
    br = java.io.BufferedReader(isr);
    serverReleaseVersion = readLine(br);
end
