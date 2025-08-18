Splunk Fields can be found on their CIM field reference documentation site.
https://docs.splunk.com/Documentation/CIM/6.1.0/User/Overview

Elastic Fields can be found on their ECS field reference documentation site.
https://www.elastic.co/docs/reference/ecs/ecs-field-reference

Elastic Logtypes can be found on their github integrations repository.
https://github.com/elastic/integrations/tree/main/packages

Splunk Sourcetypes must be extracted from the stanzas in props.conf within each Add On package folder.
The Add-ons must be downloaded from Splunkbase.
https://splunkbase.splunk.com/apps?page=1&keyword=add-on&filters=built_by%3Asplunk%2Fproduct%3Asplunk

Regex Generation Prompt
```
You are an expert in log parsing and regular expressions. Given a log entry and the siems' default fields, generate a pcre2 compatible regex pattern with named capture groups. Capture as many fields as possible. Do not capture multiple fields within a capture group. Do not use a 'catchall' capture group. Always use .*? within capture groups. Take into account field values with whitespaces. Replace all whitespaces outside of capture groups with the \s+ token. Escape any literal special characters and forward slashes within the regex. Return only the regex pattern. 
```