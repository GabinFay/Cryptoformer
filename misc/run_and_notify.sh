#!/bin/bash

# Check if command is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <command_to_run_python_script>"
    exit 1
fi

# Run the command (Python script) and capture its output
output=$($1 2>&1)

# Check if the command exited with a non-zero exit code
if [ $? -ne 0 ]; then
    # Send an email with the output
    echo "$output" | mail -s "Script failed" gabin.fay@gmail.com
fi

########
THIS IS THE NEW CRONTAB
0 * * * * /path/to/run_and_notify.sh /home/ec2-user/Bot/.venv/bin/python /home/ec2-user/Bot/fetch_data.py >> /home/ec2-user/Bot/fetch_data.log 2>&1
5 * * * * /path/to/run_and_notify.sh /home/ec2-user/Bot/.venv/bin/python /home/ec2-user/Bot/hourly_sheets_up.py >> /home/ec2-user/Bot/hourly_sheets.log 2>&1
5 0 * * * /path/to/run_and_notify.sh /home/ec2-user/Bot/.venv/bin/python /home/ec2-user/Bot/daily_sheets_up.py >> /home/ec2-user/Bot/daily_sheets.log 2>&1
