# AI Usage Disclosure

## Tools Used
I used ChatGPT to help interpret the task requirements, review the implementation, draft documentation, and develop parts of `rawaj/audit.py` and `rawaj/metrics.py`. I used PowerShell to inspect the Git history, review project files, and run local pytest tests.

## An Incorrect or Incomplete AI Suggestion
An earlier version of `audit.py` used a different function interface and output structure from the required `audit(df, *, target, id_col, time_col)` specification. I noticed the mismatch by comparing the implementation against the original task requirements. I replaced the implementation and ran the available local tests. These tests do not establish that the original starter or hidden tests will pass.

## Work Without AI
I independently reviewed the PowerShell command outputs and compared the repository contents with the required project structure. This was necessary because the original sample CSV and some starter files were absent from the current Git history, so I could not safely assume they were available or recreate them as if they were original data.

## Verification Status
The available local tests passed at the time of the check. The original sample data, starter dependency configuration, and original public tests still need to be recovered or verified before submission.
