# Workload Analysis

This repo contains the code for launching a server for workload analysis in Python.

### PCPS.py

This contains the class for the workload analysis.
To work, it requires a threshold be set `pcps.setThreshold(value)` and pupil size at brightness incrementations be set also `pcps.setIncrements(pupil_sizes_at_incrementations)`.
Then workload can be calculated by providing pupil sizes and corresponding lumninance values.

```python
from PCPS import PCPS

threshold = 2.0
pupil_sizes_at_incrementations = [5.759033, 5.788193, 5.355408, 3.851166, 3.903122, 3.626404, 3.245605, 3.270523, 3.378021, 3.311432, 3.033432, 2.741928, 2.939499, 2.738693, 2.694229, 2.6698, 2.695648, 2.608521]

pupil_left = [...]
luminance = [...]

pcps = PCPS()
pcps.setThreshold(threshold)
pcps.setIncrements(pupil_sizes_at_incrementations)
workload = pcps.calculateWorkload(pupil_left, luminance)

print(workload)
```

#### server.py

This script will host a Flask server which exposes the following endpoints for calculating workload from a remote service.

* `/` - GET for health check (returns 'OK')
* `/threshold` - POST to field "threshold" with value (returns 'OK')
* `/increments` - POST to field "increments" with comma separated string of increments (returns 'OK')
* `/workload` - POST to fields "pupil" and "luminance" with two comma separated strings of pupil sizes and luminance values (returns workload level 0 or 1)