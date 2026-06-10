import pandas as pd
import re
log_pattern = re.compile(
    r"^(?P<Timestamp>\d{6}\s\d{6})\s+(?P<PID>[0-9]+)\s+(?P<Log_level>[A-Z]+)\s+(?P<Component>.*):\s+(?P<Message>.*)$"
)
parsed_logs = []
with open("HDFS_2K.log", "r") as f:
    for line in f:
        match = log_pattern.match(line)
        if match:
            parsed_logs.append(match.groupdict())
df = pd.DataFrame(parsed_logs)
print(df)
