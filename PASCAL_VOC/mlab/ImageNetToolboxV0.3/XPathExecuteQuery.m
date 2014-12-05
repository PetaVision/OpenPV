function result = XPathExecuteQuery(xDoc, query)
    import javax.xml.xpath.*;
    factory = XPathFactory.newInstance();
    xpath = factory.newXPath();
    expression=xpath.compile(query);
    result = expression.evaluate(xDoc, XPathConstants.NODESET);
end