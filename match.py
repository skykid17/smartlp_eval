import pcre2

log = "Text it is"
regex = r"\S+\sit"
regex = ".*"

pattern = pcre2.compile(regex)
match = pattern.match(log)
if match:
    print(match)
    print("Match found:", match.group())
    percentage = len(match.group()) / len(log) * 100
    print(f"Match percentage: {percentage:.2f}%")
else:
    print("No match found")

fullmatch = pattern.fullmatch(log)
if fullmatch:
    print(fullmatch)
    print("Full match found:", fullmatch.group())
else:
    print("No full match found")

log = "- 1117989508 2005.06.05 R36-M0-NA-C:J17-U01 2005-06-05-09.38.28.957918 R36-M0-NA-C:J17-U01 RAS KERNEL INFO generating core.1172"
regex = "(?<timestamp>\d+)\s+(?<date>\d{4}\.\d{2}\.\d{2})\s+(?<device_id>[^ ]+)\s+(?<datetime>\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d{6})\s+\k<device_id>\s+(?<process>[^ ]+)\s+(?<severity>[^ ]+)\s+(?<message>.*?)"

pattern = pcre2.compile(regex)
captured = pattern.search(log)

print(f"Captured {captured.group()}")

log = "<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System><Provider Name='Microsoft-Windows-SystemDataArchiver' Guid='{4389f802-0c4f-56d0-63c6-d77db206d237}'/><EventID>2050</EventID><Version>0</Version><Level>4</Level><Task>0</Task><Opcode>0</Opcode><Keywords>0x8000000000000000</Keywords><TimeCreated SystemTime='2025-03-12T07:17:35.258212200Z'/><EventRecordID>8361459</EventRecordID><Correlation ActivityID='{3debc1ad-5b8d-48c4-ae2c-29e8968f650e}'/><Execution ProcessID='4000' ThreadID='2692'/><Channel>Microsoft-Windows-SystemDataArchiver/Diagnostic</Channel><Computer>KELWIN2019TEST2</Computer><Security UserID='S-1-5-19'/></System><EventData><Data Name='LogString'>[counter_request::Run] Delivered values for request 0x100000000000002 (1 counters).</Data></EventData></Event>"
regex = "<Event\s+xmlns='http:\/\/schemas\.microsoft\.com\/win\/2004\/08\/events\/event'><System><Provider\s+Name='(?<ProviderName>.*?)'\s+Guid='(?<ProviderGuid>.*?)'\/><EventID>(?<EventID>.*?)<\/EventID><Version>(?<Version>.*?)<\/Version><Level>(?<Level>.*?)<\/Level><Task>(?<Task>.*?)<\/Task><Opcode>(?<Opcode>.*?)<\/Opcode><Keywords>(?<Keywords>.*?)<\/Keywords><TimeCreated\s+SystemTime='(?<SystemTime>.*?)'\/><EventRecordID>(?<EventRecordID>.*?)<\/EventRecordID><Correlation\s+ActivityID='(?<ActivityID>.*?)'\/><Execution\s+ProcessID='(?<ProcessID>.*?)'\s+ThreadID='(?<ThreadID>.*?)'\/><Channel>(?<Channel>.*?)<\/Channel><Computer>(?<Computer>.*?)<\/Computer><Security\s+UserID='(?<UserID>.*?)'\/><\/System><EventData><Data\s+Name='LogString'>(?<LogString>.*?)<\/Data><\/EventData><\/Event>"
gt_regex = ".*?<System><Provider Name='(?P<provider>[^']+)' Guid='(?P<guid>[^']+)'\/><EventID>(?P<event_id>\d+)<\/EventID>.*?<Level>(?P<level>\d+)<\/Level><Task>(?P<task>\d+)<\/Task><Opcode>(?P<opcode>\d+)<\/Opcode><Keywords>(?P<keywords>[\d\w]+)<\/Keywords><TimeCreated SystemTime='(?P<timestamp>[^']+)'\/>.*?<EventRecordID>(?P<event_record_id>\d+)<\/EventRecordID><Correlation( ActivityID='{(?P<correlation_id>[\w\d\-]+)}')?\/>.*?<Execution ProcessID='(?P<process_id>\d+)' ThreadID='(?P<thread_id>\d+)'\/>.*?<Channel>(?P<channel>[^<]+)<\/Channel>.*?<Computer>(?P<computer>[^<]+)<\/Computer>.*?(<Security UserID='(?P<user_id>[^']+)'\/>).*<EventData>(?P<event>.*)<\/EventData><\/Event>"
pattern = pcre2.compile(regex)
match = pattern.match(log)

print(f"Captured {match.group()}")
# List all captured groups
for name, value in match.groupdict().items():
    print(f"{name}: {value}")

pattern = pcre2.compile(gt_regex)
match = pattern.match(log)

print(f"Golden: Captured {match.group()}")
# List all captured groups
for name, value in match.groupdict().items():
    print(f"{name}: {value}")