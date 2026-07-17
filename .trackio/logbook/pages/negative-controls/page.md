# Negative controls


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_0861bb6c9526", "created_at": "2026-07-17T05:14:09+00:00", "title": "Hiding and bound controls"}
-->
## Controls that would falsify the certificate

The test suite checks probability normalization, exact composition coverage, and that every exact finite-class tail is below its independent union-Hoeffding bound. A high-n Renyi hiding control requires the common-event probability to exceed 0.99; it does. Configurations outside the high-separation M=32 slice are retained in the raw CSV and are not used to state the high-separation summary.


---
<!-- trackio-cell
{"type": "code", "id": "cell_91080759f3f6", "created_at": "2026-07-17T05:15:18+00:00", "title": "Unit and adversarial controls", "command": ["pytest", "-q"], "exit_code": 0, "duration_s": 1.987}
-->
````bash
$ pytest -q
````

exit 0 · 2.0s


````output
..........                                                               [100%]
10 passed in 1.72s

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_027d17f8ce3d", "created_at": "2026-07-17T05:15:18+00:00", "title": "Artifact: rare_events.csv", "path": "outputs/rare_events.csv", "size": 2297, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/rare_events.csv` · dataset · 2.3 kB

trackio-local-path://outputs/rare_events.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_902ad3b95391", "created_at": "2026-07-17T05:15:18+00:00", "title": "Artifact: weak_ipm.csv", "path": "outputs/weak_ipm.csv", "size": 859, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/weak_ipm.csv` · dataset · 859 B

trackio-local-path://outputs/weak_ipm.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_33fbd67ea41d", "created_at": "2026-07-17T05:15:18+00:00", "title": "Artifact: strong_ipm.csv", "path": "outputs/strong_ipm.csv", "size": 335, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/strong_ipm.csv` · dataset · 335 B

trackio-local-path://outputs/strong_ipm.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_b043059bc11f", "created_at": "2026-07-17T05:15:44+00:00", "title": "Unit and adversarial controls", "command": ["pytest", "-q"], "exit_code": 0, "duration_s": 1.484}
-->
````bash
$ pytest -q
````

exit 0 · 1.5s


````output
..........                                                               [100%]
10 passed in 1.23s

````
