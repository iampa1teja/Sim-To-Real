# Background policy worker: CPU control-loop results

Status: complete. 36/36 new cases. [Audit](async_audit.json).

The new runs use the production async RemotePolicy, original np.save/msgpack ZMQ transport, fixed zero actions, and 50/150 ms artificial inference sleeps. Each run schedules 1,800 ticks over 60 real seconds. This measures client scheduling and CPU/loopback overhead, not learned task success, GPU latency or physical smoothness. [Commands and implementation](README.md).

The 216 reference summaries come unchanged from benchmark commit `e1b941727dbe51c234c136df72f6c1a0e3723aee` ([published report](https://github.com/iampa1teja/Sim-To-Real/blob/e1b941727dbe51c234c136df72f6c1a0e3723aee/tools/bench_transport/REPORT.md)). They are a prior serial run on this workstation, not concurrent paired trials. New hardware/software and the measured runtime commit are recorded in [environment](async_environment.json); the audit rejects runtime drift after collection.

## What changed

RemotePolicy now has an opt-in single-owner socket worker. Control ticks poll completed requests and continue dispatching buffered actions. Refill uses remaining control slots and max(configured refill, recent turnaround p95 + one period), with one inference in flight. Copied observation snapshots carry generation, monotonic capture time and control step; arrivals trim elapsed repeated steps and replace the aligned buffer.

Starvation repeats the applied, bounded target and faults after grace; request deadlines latch faults without waiting on the network. Reset, Stop, instruction change and faults invalidate older replies. Optional blending/EMA, role-based gripper exclusions, grasp feedback and seed acknowledgements remain available. Real callers keep range clamps, first-fresh-target checks and motion locks. Simulator defaults remain deterministic.

An initial pilot exposed a one-period refill delay caused by sub-period capture jitter. The accepted matrix is remeasured after correcting the control-slot boundary. Clock tests also verify that a long control pause expires targets and does not restart starvation grace. Pilot data is ignored and is not mixed with these summaries.

## Target checks

36/36 collected runs meet the single-digit hold target after startup; 0 steady-state held ticks in total. Startup holds range from 2 to 5 ticks.

Age is compared separately at p50, p95, p99 and max against the matched LeRobot raw/default reference (its raw encoding is the primary comparison). Differences are computed per payload/delay; no pooled percentile is claimed.

| Age statistic | New cells no worse than reference |
|---|---:|
| p50 | 36/36 |
| p95 | 36/36 |
| p99 | 36/36 |
| max | 36/36 |

## Side-by-side measurements

Held includes startup. Steady held is available only for the new runs; reference raw traces are retained on the benchmark branch. Age includes retained targets during holds and excludes startup slots before the first prediction. Turnaround is observation capture through completion of the worker request, including input copies/seed RPC when enabled; the old B summaries did not record a turnaround distribution. Replacements count accepted chunks after the first inside the window; new chunks activate on the control tick. All times are milliseconds.

| H×W / cameras / chunk | Delay | Setup | Held / 1800 | Steady held | Age p50 | Age p95 | Age p99 | Age max | Replacements | Turnaround p50 / p95 / p99 / max |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 480×640 / 3 / 50 | 150 | lerobot jpeg default | 6 | — | 767.952 | 1268.096 | 1302.319 | 1534.972 | 52 | — |
| 480×640 / 3 / 50 | 150 | lerobot jpeg tuned | 6 | — | 901.238 | 1534.699 | 1568.908 | 1801.415 | 42 | — |
| 480×640 / 3 / 50 | 150 | lerobot raw default | 6 | — | 734.986 | 1268.024 | 1301.484 | 1501.447 | 52 | — |
| 480×640 / 3 / 50 | 150 | lerobot raw tuned | 6 | — | 901.379 | 1534.815 | 1601.484 | 1801.526 | 41 | — |
| 480×640 / 3 / 50 | 150 | zmq prefetch | 157 | — | 1166.798 | 1866.861 | 1933.740 | 1967.277 | 38 | — |
| 480×640 / 3 / 50 | 150 | zmq sync | 137 | — | 1000.178 | 1833.179 | 1899.856 | 1933.277 | 33 | — |
| 480×640 / 3 / 50 | 150 | **zmq async** | 5 | 0 | 700.379 | 1233.388 | 1266.833 | 1300.619 | 52 | 159.067 / 160.853 / 168.294 / 173.864 |
| 480×640 / 3 / 50 | 50 | lerobot jpeg default | 3 | — | 668.032 | 1168.256 | 1207.580 | 1334.901 | 52 | — |
| 480×640 / 3 / 50 | 50 | lerobot jpeg tuned | 3 | — | 834.814 | 1534.442 | 1601.601 | 1701.708 | 39 | — |
| 480×640 / 3 / 50 | 50 | lerobot raw default | 3 | — | 635.153 | 1168.318 | 1201.599 | 1301.550 | 52 | — |
| 480×640 / 3 / 50 | 50 | lerobot raw tuned | 2 | — | 835.846 | 1534.841 | 1601.550 | 1668.341 | 39 | — |
| 480×640 / 3 / 50 | 50 | zmq prefetch | 40 | — | 966.761 | 1666.802 | 1733.741 | 1767.232 | 38 | — |
| 480×640 / 3 / 50 | 50 | zmq sync | 37 | — | 866.754 | 1633.422 | 1699.916 | 1733.277 | 35 | — |
| 480×640 / 3 / 50 | 50 | **zmq async** | 2 | 0 | 600.496 | 1134.028 | 1166.876 | 1200.608 | 52 | 58.999 / 60.144 / 66.158 / 72.074 |
| 480×640 / 3 / 30 | 150 | lerobot jpeg default | 6 | — | 434.590 | 634.736 | 637.280 | 834.946 | 127 | — |
| 480×640 / 3 / 30 | 150 | lerobot jpeg tuned | 6 | — | 567.996 | 901.343 | 934.593 | 1168.125 | 81 | — |
| 480×640 / 3 / 30 | 150 | lerobot raw default | 6 | — | 434.423 | 634.697 | 634.904 | 834.868 | 127 | — |
| 480×640 / 3 / 30 | 150 | lerobot raw tuned | 6 | — | 568.072 | 901.492 | 934.834 | 1134.713 | 77 | — |
| 480×640 / 3 / 30 | 150 | zmq prefetch | 269 | — | 833.411 | 1233.476 | 1267.057 | 1300.496 | 66 | — |
| 480×640 / 3 / 30 | 150 | zmq sync | 213 | — | 666.855 | 1199.844 | 1233.252 | 1266.609 | 52 | — |
| 480×640 / 3 / 30 | 150 | **zmq async** | 5 | 0 | 367.290 | 600.055 | 600.241 | 633.926 | 128 | 158.968 / 161.141 / 162.394 / 173.612 |
| 480×640 / 3 / 30 | 50 | lerobot jpeg default | 3 | — | 334.599 | 534.958 | 568.133 | 635.046 | 128 | — |
| 480×640 / 3 / 30 | 50 | lerobot jpeg tuned | 3 | — | 501.553 | 900.579 | 934.858 | 1069.673 | 71 | — |
| 480×640 / 3 / 30 | 50 | lerobot raw default | 3 | — | 333.869 | 534.836 | 535.014 | 635.015 | 128 | — |
| 480×640 / 3 / 30 | 50 | lerobot raw tuned | 3 | — | 533.748 | 901.752 | 934.960 | 1035.107 | 69 | — |
| 480×640 / 3 / 30 | 50 | zmq prefetch | 68 | — | 633.439 | 1033.484 | 1067.040 | 1100.550 | 66 | — |
| 480×640 / 3 / 30 | 50 | zmq sync | 60 | — | 533.419 | 1000.086 | 1033.266 | 1066.611 | 57 | — |
| 480×640 / 3 / 30 | 50 | **zmq async** | 2 | 0 | 267.101 | 500.096 | 500.318 | 533.948 | 128 | 58.955 / 60.923 / 63.499 / 74.479 |
| 480×640 / 3 / 16 | 150 | lerobot jpeg default | 6 | — | 301.559 | 401.366 | 401.570 | 434.894 | 314 | — |
| 480×640 / 3 / 16 | 150 | lerobot jpeg tuned | 6 | — | 334.848 | 434.826 | 468.154 | 701.369 | 223 | — |
| 480×640 / 3 / 16 | 150 | lerobot raw default | 6 | — | 268.183 | 367.924 | 368.183 | 401.437 | 348 | — |
| 480×640 / 3 / 16 | 150 | lerobot raw tuned | 6 | — | 335.010 | 468.074 | 468.242 | 668.131 | 198 | — |
| 480×640 / 3 / 16 | 150 | zmq prefetch | 557 | — | 600.118 | 800.294 | 800.398 | 833.914 | 137 | — |
| 480×640 / 3 / 16 | 150 | zmq sync | 361 | — | 433.526 | 766.466 | 766.599 | 799.944 | 89 | — |
| 480×640 / 3 / 16 | 150 | **zmq async** | 5 | 0 | 233.203 | 299.949 | 300.106 | 334.034 | 358 | 158.758 / 160.540 / 163.365 / 174.039 |
| 480×640 / 3 / 16 | 50 | lerobot jpeg default | 3 | — | 168.089 | 202.107 | 202.470 | 235.000 | 653 | — |
| 480×640 / 3 / 16 | 50 | lerobot jpeg tuned | 3 | — | 272.729 | 437.347 | 470.494 | 570.563 | 160 | — |
| 480×640 / 3 / 16 | 50 | lerobot raw default | 3 | — | 134.838 | 168.234 | 168.435 | 201.556 | 837 | — |
| 480×640 / 3 / 16 | 50 | lerobot raw tuned | 3 | — | 268.211 | 468.163 | 468.350 | 568.411 | 149 | — |
| 480×640 / 3 / 16 | 50 | zmq prefetch | 139 | — | 400.122 | 600.282 | 600.405 | 600.596 | 138 | — |
| 480×640 / 3 / 16 | 50 | zmq sync | 107 | — | 300.093 | 566.506 | 566.583 | 599.944 | 105 | — |
| 480×640 / 3 / 16 | 50 | **zmq async** | 2 | 0 | 99.329 | 100.020 | 100.366 | 133.904 | 898 | 58.589 / 60.125 / 62.183 / 72.530 |
| 480×640 / 2 / 50 | 150 | lerobot jpeg default | 6 | — | 734.598 | 1267.716 | 1301.136 | 1501.170 | 52 | — |
| 480×640 / 2 / 50 | 150 | lerobot jpeg tuned | 6 | — | 901.014 | 1534.601 | 1601.134 | 1801.130 | 41 | — |
| 480×640 / 2 / 50 | 150 | lerobot raw default | 5 | — | 734.482 | 1267.624 | 1301.196 | 1467.824 | 52 | — |
| 480×640 / 2 / 50 | 150 | lerobot raw tuned | 5 | — | 901.125 | 1534.565 | 1601.130 | 1767.804 | 41 | — |
| 480×640 / 2 / 50 | 150 | zmq prefetch | 156 | — | 1166.754 | 1866.824 | 1933.724 | 1933.852 | 38 | — |
| 480×640 / 2 / 50 | 150 | zmq sync | 136 | — | 1000.163 | 1833.181 | 1899.854 | 1899.943 | 33 | — |
| 480×640 / 2 / 50 | 150 | **zmq async** | 5 | 0 | 700.340 | 1233.358 | 1266.780 | 1267.125 | 52 | 161.098 / 162.122 / 163.772 / 163.865 |
| 480×640 / 2 / 50 | 50 | lerobot jpeg default | 3 | — | 634.811 | 1167.559 | 1202.721 | 1301.313 | 52 | — |
| 480×640 / 2 / 50 | 50 | lerobot jpeg tuned | 3 | — | 867.456 | 1534.535 | 1601.298 | 1701.261 | 39 | — |
| 480×640 / 2 / 50 | 50 | lerobot raw default | 2 | — | 634.483 | 1134.706 | 1201.178 | 1267.892 | 52 | — |
| 480×640 / 2 / 50 | 50 | lerobot raw tuned | 2 | — | 851.090 | 1534.511 | 1601.304 | 1667.680 | 39 | — |
| 480×640 / 2 / 50 | 50 | zmq prefetch | 39 | — | 966.785 | 1666.826 | 1733.742 | 1733.861 | 38 | — |
| 480×640 / 2 / 50 | 50 | zmq sync | 36 | — | 866.699 | 1633.380 | 1699.918 | 1699.944 | 35 | — |
| 480×640 / 2 / 50 | 50 | **zmq async** | 2 | 0 | 600.379 | 1133.836 | 1166.834 | 1167.405 | 52 | 61.089 / 62.703 / 64.844 / 65.275 |
| 480×640 / 2 / 30 | 150 | lerobot jpeg default | 6 | — | 401.788 | 634.393 | 634.562 | 834.543 | 127 | — |
| 480×640 / 2 / 30 | 150 | lerobot jpeg tuned | 6 | — | 567.795 | 901.179 | 934.520 | 1134.553 | 77 | — |
| 480×640 / 2 / 30 | 150 | lerobot raw default | 5 | — | 401.115 | 601.218 | 635.891 | 801.158 | 127 | — |
| 480×640 / 2 / 30 | 150 | lerobot raw tuned | 5 | — | 567.794 | 901.197 | 934.514 | 1101.192 | 77 | — |
| 480×640 / 2 / 30 | 150 | zmq prefetch | 268 | — | 833.419 | 1233.488 | 1267.033 | 1267.266 | 66 | — |
| 480×640 / 2 / 30 | 150 | zmq sync | 212 | — | 666.850 | 1199.845 | 1233.252 | 1233.277 | 52 | — |
| 480×640 / 2 / 30 | 150 | **zmq async** | 5 | 0 | 367.101 | 600.012 | 600.206 | 600.631 | 128 | 156.781 / 157.642 / 158.692 / 165.044 |
| 480×640 / 2 / 30 | 50 | lerobot jpeg default | 3 | — | 334.156 | 534.511 | 534.673 | 634.691 | 128 | — |
| 480×640 / 2 / 30 | 50 | lerobot jpeg tuned | 3 | — | 534.148 | 901.113 | 934.635 | 1034.530 | 69 | — |
| 480×640 / 2 / 30 | 50 | lerobot raw default | 2 | — | 301.157 | 501.369 | 534.661 | 601.265 | 128 | — |
| 480×640 / 2 / 30 | 50 | lerobot raw tuned | 2 | — | 517.664 | 901.345 | 934.581 | 1001.210 | 69 | — |
| 480×640 / 2 / 30 | 50 | zmq prefetch | 67 | — | 633.448 | 1033.495 | 1067.072 | 1067.211 | 66 | — |
| 480×640 / 2 / 30 | 50 | zmq sync | 59 | — | 533.395 | 1000.049 | 1033.264 | 1033.277 | 58 | — |
| 480×640 / 2 / 30 | 50 | **zmq async** | 2 | 0 | 267.120 | 500.090 | 500.279 | 500.570 | 128 | 56.765 / 58.628 / 59.534 / 65.058 |
| 480×640 / 2 / 16 | 150 | lerobot jpeg default | 6 | — | 301.072 | 367.873 | 401.151 | 403.213 | 323 | — |
| 480×640 / 2 / 16 | 150 | lerobot jpeg tuned | 6 | — | 334.259 | 467.773 | 467.890 | 667.796 | 198 | — |
| 480×640 / 2 / 16 | 150 | lerobot raw default | 6 | — | 267.787 | 334.500 | 334.599 | 401.324 | 358 | — |
| 480×640 / 2 / 16 | 150 | lerobot raw tuned | 6 | — | 334.618 | 467.806 | 467.947 | 668.345 | 198 | — |
| 480×640 / 2 / 16 | 150 | zmq prefetch | 556 | — | 600.062 | 800.296 | 800.407 | 800.500 | 137 | — |
| 480×640 / 2 / 16 | 150 | zmq sync | 360 | — | 433.520 | 733.271 | 766.590 | 766.610 | 89 | — |
| 480×640 / 2 / 16 | 150 | **zmq async** | 5 | 0 | 233.205 | 299.941 | 300.129 | 300.603 | 358 | 156.610 / 157.520 / 158.690 / 163.956 |
| 480×640 / 2 / 16 | 50 | lerobot jpeg default | 3 | — | 136.749 | 201.137 | 201.306 | 204.689 | 691 | — |
| 480×640 / 2 / 16 | 50 | lerobot jpeg tuned | 3 | — | 268.321 | 467.873 | 468.007 | 569.259 | 149 | — |
| 480×640 / 2 / 16 | 50 | lerobot raw default | 3 | — | 101.799 | 134.688 | 140.070 | 202.930 | 902 | — |
| 480×640 / 2 / 16 | 50 | lerobot raw tuned | 2 | — | 268.167 | 467.829 | 467.954 | 534.566 | 149 | — |
| 480×640 / 2 / 16 | 50 | zmq prefetch | 139 | — | 400.124 | 600.311 | 600.426 | 600.563 | 138 | — |
| 480×640 / 2 / 16 | 50 | zmq sync | 106 | — | 300.094 | 566.513 | 566.595 | 566.611 | 105 | — |
| 480×640 / 2 / 16 | 50 | **zmq async** | 2 | 0 | 84.541 | 100.018 | 100.289 | 100.689 | 898 | 56.445 / 58.064 / 59.169 / 63.156 |
| 480×640 / 1 / 50 | 150 | lerobot jpeg default | 6 | — | 734.374 | 1267.424 | 1300.848 | 1500.634 | 52 | — |
| 480×640 / 1 / 50 | 150 | lerobot jpeg tuned | 6 | — | 900.741 | 1534.232 | 1600.831 | 1801.054 | 41 | — |
| 480×640 / 1 / 50 | 150 | lerobot raw default | 5 | — | 700.941 | 1234.082 | 1267.506 | 1433.964 | 52 | — |
| 480×640 / 1 / 50 | 150 | lerobot raw tuned | 5 | — | 867.385 | 1500.928 | 1567.494 | 1767.516 | 41 | — |
| 480×640 / 1 / 50 | 150 | zmq prefetch | 156 | — | 1166.777 | 1866.806 | 1933.746 | 1934.053 | 38 | — |
| 480×640 / 1 / 50 | 150 | zmq sync | 136 | — | 1000.160 | 1833.181 | 1899.860 | 1899.944 | 33 | — |
| 480×640 / 1 / 50 | 150 | **zmq async** | 5 | 0 | 700.316 | 1233.356 | 1266.843 | 1267.326 | 52 | 154.870 / 156.105 / 157.871 / 158.072 |
| 480×640 / 1 / 50 | 50 | lerobot jpeg default | 3 | — | 634.446 | 1167.499 | 1200.853 | 1300.723 | 52 | — |
| 480×640 / 1 / 50 | 50 | lerobot jpeg tuned | 3 | — | 867.180 | 1534.216 | 1600.838 | 1700.947 | 39 | — |
| 480×640 / 1 / 50 | 50 | lerobot raw default | 2 | — | 617.508 | 1134.080 | 1167.599 | 1267.430 | 52 | — |
| 480×640 / 1 / 50 | 50 | lerobot raw tuned | 2 | — | 833.810 | 1500.834 | 1567.549 | 1667.413 | 39 | — |
| 480×640 / 1 / 50 | 50 | zmq prefetch | 39 | — | 966.766 | 1666.827 | 1733.732 | 1733.926 | 38 | — |
| 480×640 / 1 / 50 | 50 | zmq sync | 36 | — | 866.684 | 1633.378 | 1699.921 | 1699.944 | 35 | — |
| 480×640 / 1 / 50 | 50 | **zmq async** | 2 | 0 | 600.419 | 1133.602 | 1166.844 | 1167.233 | 52 | 54.673 / 55.536 / 58.676 / 59.331 |
| 480×640 / 1 / 30 | 150 | lerobot jpeg default | 6 | — | 401.097 | 634.073 | 634.224 | 800.847 | 127 | — |
| 480×640 / 1 / 30 | 150 | lerobot jpeg tuned | 6 | — | 567.430 | 900.870 | 934.195 | 1134.114 | 77 | — |
| 480×640 / 1 / 30 | 150 | lerobot raw default | 5 | — | 400.537 | 600.751 | 600.901 | 800.842 | 127 | — |
| 480×640 / 1 / 30 | 150 | lerobot raw tuned | 5 | — | 534.142 | 867.560 | 900.843 | 1102.398 | 77 | — |
| 480×640 / 1 / 30 | 150 | zmq prefetch | 268 | — | 833.446 | 1233.486 | 1267.041 | 1267.151 | 66 | — |
| 480×640 / 1 / 30 | 150 | zmq sync | 212 | — | 666.856 | 1199.847 | 1233.240 | 1233.277 | 52 | — |
| 480×640 / 1 / 30 | 150 | **zmq async** | 5 | 0 | 367.051 | 600.018 | 600.230 | 600.610 | 128 | 154.824 / 156.274 / 157.000 / 159.868 |
| 480×640 / 1 / 30 | 50 | lerobot jpeg default | 3 | — | 333.476 | 534.116 | 534.244 | 634.245 | 128 | — |
| 480×640 / 1 / 30 | 50 | lerobot jpeg tuned | 3 | — | 501.383 | 900.929 | 934.217 | 1034.157 | 69 | — |
| 480×640 / 1 / 30 | 50 | lerobot raw default | 2 | — | 284.471 | 500.827 | 500.986 | 600.893 | 128 | — |
| 480×640 / 1 / 30 | 50 | lerobot raw tuned | 2 | — | 500.297 | 867.520 | 900.950 | 1000.844 | 69 | — |
| 480×640 / 1 / 30 | 50 | zmq prefetch | 67 | — | 633.405 | 1033.480 | 1067.092 | 1067.324 | 66 | — |
| 480×640 / 1 / 30 | 50 | zmq sync | 59 | — | 533.416 | 1000.043 | 1033.266 | 1033.296 | 58 | — |
| 480×640 / 1 / 30 | 50 | **zmq async** | 2 | 0 | 267.100 | 500.074 | 500.309 | 500.706 | 128 | 54.802 / 56.391 / 57.904 / 60.713 |
| 480×640 / 1 / 16 | 150 | lerobot jpeg default | 6 | — | 267.549 | 367.411 | 367.536 | 400.892 | 342 | — |
| 480×640 / 1 / 16 | 150 | lerobot jpeg tuned | 6 | — | 334.221 | 467.470 | 467.574 | 667.500 | 198 | — |
| 480×640 / 1 / 16 | 150 | lerobot raw default | 5 | — | 267.406 | 334.144 | 334.256 | 367.526 | 369 | — |
| 480×640 / 1 / 16 | 150 | lerobot raw tuned | 5 | — | 300.756 | 434.106 | 434.239 | 600.900 | 198 | — |
| 480×640 / 1 / 16 | 150 | zmq prefetch | 556 | — | 600.091 | 800.311 | 800.402 | 800.486 | 137 | — |
| 480×640 / 1 / 16 | 150 | zmq sync | 360 | — | 433.560 | 733.284 | 766.591 | 766.626 | 89 | — |
| 480×640 / 1 / 16 | 150 | **zmq async** | 5 | 0 | 233.208 | 299.939 | 300.092 | 300.675 | 358 | 154.456 / 155.512 / 156.201 / 160.621 |
| 480×640 / 1 / 16 | 50 | lerobot jpeg default | 3 | — | 134.136 | 167.549 | 167.696 | 200.620 | 791 | — |
| 480×640 / 1 / 16 | 50 | lerobot jpeg tuned | 3 | — | 267.830 | 467.473 | 467.573 | 567.638 | 149 | — |
| 480×640 / 1 / 16 | 50 | lerobot raw default | 2 | — | 100.895 | 134.236 | 134.325 | 167.471 | 976 | — |
| 480×640 / 1 / 16 | 50 | lerobot raw tuned | 2 | — | 267.158 | 434.198 | 434.313 | 534.173 | 149 | — |
| 480×640 / 1 / 16 | 50 | zmq prefetch | 139 | — | 400.126 | 600.313 | 600.423 | 600.856 | 138 | — |
| 480×640 / 1 / 16 | 50 | zmq sync | 106 | — | 300.087 | 566.512 | 566.598 | 566.634 | 105 | — |
| 480×640 / 1 / 16 | 50 | **zmq async** | 2 | 0 | 83.990 | 100.046 | 100.306 | 100.636 | 898 | 54.371 / 55.315 / 55.944 / 58.643 |
| 224×224 / 3 / 50 | 150 | lerobot jpeg default | 6 | — | 734.052 | 1267.236 | 1300.651 | 1500.700 | 52 | — |
| 224×224 / 3 / 50 | 150 | lerobot jpeg tuned | 6 | — | 900.552 | 1534.066 | 1600.698 | 1800.614 | 41 | — |
| 224×224 / 3 / 50 | 150 | lerobot raw default | 5 | — | 700.863 | 1233.924 | 1267.330 | 1433.964 | 52 | — |
| 224×224 / 3 / 50 | 150 | lerobot raw tuned | 5 | — | 867.201 | 1500.729 | 1567.332 | 1734.047 | 41 | — |
| 224×224 / 3 / 50 | 150 | zmq prefetch | 156 | — | 1166.736 | 1866.790 | 1933.718 | 1933.930 | 38 | — |
| 224×224 / 3 / 50 | 150 | zmq sync | 136 | — | 1000.107 | 1833.176 | 1899.919 | 1899.944 | 33 | — |
| 224×224 / 3 / 50 | 150 | **zmq async** | 5 | 0 | 700.266 | 1233.342 | 1266.801 | 1267.309 | 52 | 153.714 / 154.103 / 155.455 / 156.031 |
| 224×224 / 3 / 50 | 50 | lerobot jpeg default | 3 | — | 634.109 | 1167.195 | 1200.698 | 1300.746 | 52 | — |
| 224×224 / 3 / 50 | 50 | lerobot jpeg tuned | 3 | — | 834.193 | 1534.031 | 1600.685 | 1700.710 | 39 | — |
| 224×224 / 3 / 50 | 50 | lerobot raw default | 2 | — | 600.886 | 1133.829 | 1167.339 | 1234.097 | 52 | — |
| 224×224 / 3 / 50 | 50 | lerobot raw tuned | 2 | — | 817.403 | 1500.658 | 1567.355 | 1634.005 | 39 | — |
| 224×224 / 3 / 50 | 50 | zmq prefetch | 39 | — | 966.740 | 1666.831 | 1733.713 | 1733.859 | 38 | — |
| 224×224 / 3 / 50 | 50 | zmq sync | 36 | — | 866.747 | 1633.415 | 1699.855 | 1699.944 | 35 | — |
| 224×224 / 3 / 50 | 50 | **zmq async** | 2 | 0 | 600.438 | 1133.477 | 1166.842 | 1167.201 | 52 | 53.752 / 54.823 / 55.385 / 55.844 |
| 224×224 / 3 / 30 | 150 | lerobot jpeg default | 6 | — | 400.896 | 633.883 | 634.046 | 834.060 | 127 | — |
| 224×224 / 3 / 30 | 150 | lerobot jpeg tuned | 5 | — | 567.216 | 900.664 | 934.071 | 1100.489 | 77 | — |
| 224×224 / 3 / 30 | 150 | lerobot raw default | 5 | — | 400.329 | 600.561 | 600.723 | 767.375 | 127 | — |
| 224×224 / 3 / 30 | 150 | lerobot raw tuned | 5 | — | 533.989 | 867.382 | 900.697 | 1067.445 | 77 | — |
| 224×224 / 3 / 30 | 150 | zmq prefetch | 268 | — | 833.379 | 1233.465 | 1267.042 | 1267.171 | 66 | — |
| 224×224 / 3 / 30 | 150 | zmq sync | 212 | — | 666.863 | 1199.842 | 1233.253 | 1233.277 | 52 | — |
| 224×224 / 3 / 30 | 150 | **zmq async** | 5 | 0 | 367.349 | 600.012 | 600.225 | 600.743 | 128 | 153.775 / 155.055 / 155.762 / 156.206 |
| 224×224 / 3 / 30 | 50 | lerobot jpeg default | 2 | — | 301.039 | 533.933 | 534.110 | 600.722 | 128 | — |
| 224×224 / 3 / 30 | 50 | lerobot jpeg tuned | 2 | — | 500.840 | 900.688 | 934.035 | 967.475 | 69 | — |
| 224×224 / 3 / 30 | 50 | lerobot raw default | 2 | — | 268.057 | 500.593 | 500.722 | 567.418 | 128 | — |
| 224×224 / 3 / 30 | 50 | lerobot raw tuned | 2 | — | 483.888 | 867.370 | 900.713 | 967.485 | 69 | — |
| 224×224 / 3 / 30 | 50 | zmq prefetch | 67 | — | 633.433 | 1033.528 | 1067.087 | 1067.184 | 66 | — |
| 224×224 / 3 / 30 | 50 | zmq sync | 59 | — | 533.424 | 1000.061 | 1033.256 | 1033.277 | 58 | — |
| 224×224 / 3 / 30 | 50 | **zmq async** | 2 | 0 | 267.039 | 500.047 | 500.208 | 500.680 | 128 | 53.631 / 54.334 / 55.464 / 56.079 |
| 224×224 / 3 / 16 | 150 | lerobot jpeg default | 6 | — | 267.292 | 334.025 | 367.248 | 400.756 | 356 | — |
| 224×224 / 3 / 16 | 150 | lerobot jpeg tuned | 6 | — | 333.962 | 467.291 | 467.618 | 667.297 | 198 | — |
| 224×224 / 3 / 16 | 150 | lerobot raw default | 5 | — | 267.199 | 333.960 | 334.051 | 337.669 | 374 | — |
| 224×224 / 3 / 16 | 150 | lerobot raw tuned | 5 | — | 300.624 | 433.997 | 434.055 | 600.605 | 198 | — |
| 224×224 / 3 / 16 | 150 | zmq prefetch | 556 | — | 600.120 | 800.309 | 800.415 | 800.703 | 137 | — |
| 224×224 / 3 / 16 | 150 | zmq sync | 360 | — | 433.526 | 733.283 | 766.602 | 766.617 | 89 | — |
| 224×224 / 3 / 16 | 150 | **zmq async** | 5 | 0 | 233.213 | 299.951 | 300.154 | 300.442 | 358 | 153.531 / 154.224 / 154.913 / 155.679 |
| 224×224 / 3 / 16 | 50 | lerobot jpeg default | 2 | — | 133.702 | 134.187 | 167.351 | 170.018 | 881 | — |
| 224×224 / 3 / 16 | 50 | lerobot jpeg tuned | 2 | — | 267.444 | 467.263 | 467.477 | 533.924 | 149 | — |
| 224×224 / 3 / 16 | 50 | lerobot raw default | 2 | — | 100.653 | 134.044 | 134.175 | 137.526 | 1005 | — |
| 224×224 / 3 / 16 | 50 | lerobot raw tuned | 2 | — | 235.248 | 433.958 | 434.062 | 500.670 | 149 | — |
| 224×224 / 3 / 16 | 50 | zmq prefetch | 139 | — | 400.119 | 600.307 | 600.394 | 600.481 | 138 | — |
| 224×224 / 3 / 16 | 50 | zmq sync | 106 | — | 300.095 | 566.513 | 566.600 | 566.611 | 105 | — |
| 224×224 / 3 / 16 | 50 | **zmq async** | 2 | 0 | 83.854 | 99.994 | 100.246 | 100.709 | 898 | 53.379 / 54.313 / 55.041 / 55.925 |
| 224×224 / 2 / 50 | 150 | lerobot jpeg default | 6 | — | 733.752 | 1233.919 | 1300.436 | 1467.329 | 52 | — |
| 224×224 / 2 / 50 | 150 | lerobot jpeg tuned | 5 | — | 867.225 | 1533.771 | 1567.309 | 1767.325 | 41 | — |
| 224×224 / 2 / 50 | 150 | lerobot raw default | 5 | — | 700.743 | 1233.885 | 1267.299 | 1433.975 | 52 | — |
| 224×224 / 2 / 50 | 150 | lerobot raw tuned | 5 | — | 867.185 | 1500.663 | 1567.236 | 1734.009 | 41 | — |
| 224×224 / 2 / 50 | 150 | zmq prefetch | 156 | — | 1166.734 | 1866.803 | 1933.728 | 1933.933 | 38 | — |
| 224×224 / 2 / 50 | 150 | zmq sync | 136 | — | 1000.163 | 1833.180 | 1899.922 | 1899.944 | 33 | — |
| 224×224 / 2 / 50 | 150 | **zmq async** | 5 | 0 | 700.304 | 1233.365 | 1266.808 | 1267.282 | 52 | 153.538 / 154.393 / 155.939 / 156.220 |
| 224×224 / 2 / 50 | 50 | lerobot jpeg default | 3 | — | 633.659 | 1133.913 | 1167.351 | 1300.385 | 52 | — |
| 224×224 / 2 / 50 | 50 | lerobot jpeg tuned | 3 | — | 833.696 | 1500.793 | 1567.299 | 1700.554 | 39 | — |
| 224×224 / 2 / 50 | 50 | lerobot raw default | 2 | — | 600.978 | 1133.744 | 1167.301 | 1233.939 | 52 | — |
| 224×224 / 2 / 50 | 50 | lerobot raw tuned | 2 | — | 817.108 | 1500.634 | 1567.272 | 1634.030 | 39 | — |
| 224×224 / 2 / 50 | 50 | zmq prefetch | 39 | — | 966.704 | 1666.809 | 1733.719 | 1733.875 | 38 | — |
| 224×224 / 2 / 50 | 50 | zmq sync | 36 | — | 866.682 | 1633.383 | 1699.910 | 1699.963 | 35 | — |
| 224×224 / 2 / 50 | 50 | **zmq async** | 2 | 0 | 600.384 | 1133.431 | 1166.884 | 1167.274 | 52 | 53.341 / 54.034 / 55.279 / 55.749 |
| 224×224 / 2 / 30 | 150 | lerobot jpeg default | 5 | — | 400.214 | 600.535 | 600.765 | 800.494 | 127 | — |
| 224×224 / 2 / 30 | 150 | lerobot jpeg tuned | 6 | — | 533.945 | 900.170 | 900.696 | 1100.713 | 77 | — |
| 224×224 / 2 / 30 | 150 | lerobot raw default | 5 | — | 400.142 | 600.541 | 600.662 | 767.298 | 127 | — |
| 224×224 / 2 / 30 | 150 | lerobot raw tuned | 5 | — | 533.886 | 867.326 | 900.655 | 1067.215 | 77 | — |
| 224×224 / 2 / 30 | 150 | zmq prefetch | 268 | — | 833.381 | 1233.472 | 1267.077 | 1267.310 | 66 | — |
| 224×224 / 2 / 30 | 150 | zmq sync | 212 | — | 666.844 | 1199.847 | 1233.259 | 1233.276 | 52 | — |
| 224×224 / 2 / 30 | 150 | **zmq async** | 5 | 0 | 367.209 | 600.041 | 600.254 | 600.561 | 128 | 153.366 / 154.412 / 155.096 / 156.303 |
| 224×224 / 2 / 30 | 50 | lerobot jpeg default | 3 | — | 300.404 | 500.601 | 533.947 | 633.871 | 128 | — |
| 224×224 / 2 / 30 | 50 | lerobot jpeg tuned | 3 | — | 500.395 | 867.479 | 900.692 | 1000.598 | 69 | — |
| 224×224 / 2 / 30 | 50 | lerobot raw default | 2 | — | 267.676 | 500.541 | 500.662 | 567.207 | 128 | — |
| 224×224 / 2 / 30 | 50 | lerobot raw tuned | 2 | — | 483.790 | 867.318 | 900.622 | 967.385 | 69 | — |
| 224×224 / 2 / 30 | 50 | zmq prefetch | 67 | — | 633.437 | 1033.454 | 1067.021 | 1067.137 | 66 | — |
| 224×224 / 2 / 30 | 50 | zmq sync | 59 | — | 533.437 | 1000.097 | 1033.234 | 1033.276 | 58 | — |
| 224×224 / 2 / 30 | 50 | **zmq async** | 2 | 0 | 267.040 | 500.051 | 500.218 | 500.496 | 128 | 53.286 / 53.704 / 54.548 / 55.514 |
| 224×224 / 2 / 16 | 150 | lerobot jpeg default | 6 | — | 267.192 | 333.948 | 334.017 | 400.610 | 362 | — |
| 224×224 / 2 / 16 | 150 | lerobot jpeg tuned | 6 | — | 300.617 | 433.958 | 467.369 | 667.371 | 198 | — |
| 224×224 / 2 / 16 | 150 | lerobot raw default | 5 | — | 267.118 | 333.901 | 333.994 | 335.960 | 376 | — |
| 224×224 / 2 / 16 | 150 | lerobot raw tuned | 5 | — | 300.579 | 433.910 | 433.982 | 600.645 | 198 | — |
| 224×224 / 2 / 16 | 150 | zmq prefetch | 556 | — | 600.117 | 800.299 | 800.392 | 800.532 | 137 | — |
| 224×224 / 2 / 16 | 150 | zmq sync | 360 | — | 433.515 | 733.307 | 766.587 | 766.645 | 89 | — |
| 224×224 / 2 / 16 | 150 | **zmq async** | 5 | 0 | 233.185 | 299.940 | 300.146 | 300.660 | 358 | 153.128 / 153.533 / 154.049 / 156.220 |
| 224×224 / 2 / 16 | 50 | lerobot jpeg default | 2 | — | 100.677 | 134.007 | 134.191 | 167.306 | 930 | — |
| 224×224 / 2 / 16 | 50 | lerobot jpeg tuned | 2 | — | 266.985 | 433.911 | 434.024 | 533.844 | 149 | — |
| 224×224 / 2 / 16 | 50 | lerobot raw default | 2 | — | 100.589 | 133.982 | 134.062 | 137.536 | 1023 | — |
| 224×224 / 2 / 16 | 50 | lerobot raw tuned | 2 | — | 234.585 | 433.862 | 433.970 | 500.628 | 149 | — |
| 224×224 / 2 / 16 | 50 | zmq prefetch | 139 | — | 400.123 | 600.298 | 600.410 | 600.520 | 138 | — |
| 224×224 / 2 / 16 | 50 | zmq sync | 106 | — | 300.091 | 566.503 | 566.601 | 566.613 | 105 | — |
| 224×224 / 2 / 16 | 50 | **zmq async** | 2 | 0 | 83.743 | 99.997 | 100.264 | 100.728 | 898 | 53.068 / 53.876 / 54.378 / 55.073 |
| 224×224 / 1 / 50 | 150 | lerobot jpeg default | 5 | — | 700.773 | 1233.837 | 1267.243 | 1433.955 | 52 | — |
| 224×224 / 1 / 50 | 150 | lerobot jpeg tuned | 5 | — | 867.193 | 1500.706 | 1567.244 | 1733.960 | 41 | — |
| 224×224 / 1 / 50 | 150 | lerobot raw default | 5 | — | 700.773 | 1233.813 | 1267.258 | 1433.962 | 52 | — |
| 224×224 / 1 / 50 | 150 | lerobot raw tuned | 5 | — | 867.168 | 1500.623 | 1567.218 | 1733.922 | 41 | — |
| 224×224 / 1 / 50 | 150 | zmq prefetch | 156 | — | 1166.748 | 1866.818 | 1933.734 | 1933.854 | 38 | — |
| 224×224 / 1 / 50 | 150 | zmq sync | 136 | — | 1000.146 | 1833.181 | 1899.861 | 1899.987 | 33 | — |
| 224×224 / 1 / 50 | 150 | **zmq async** | 5 | 0 | 700.311 | 1233.368 | 1266.811 | 1267.538 | 52 | 152.944 / 153.425 / 154.772 / 155.389 |
| 224×224 / 1 / 50 | 50 | lerobot jpeg default | 2 | — | 600.952 | 1133.775 | 1167.237 | 1233.957 | 52 | — |
| 224×224 / 1 / 50 | 50 | lerobot jpeg tuned | 3 | — | 833.553 | 1500.582 | 1567.191 | 1667.205 | 39 | — |
| 224×224 / 1 / 50 | 50 | lerobot raw default | 2 | — | 600.892 | 1133.787 | 1167.245 | 1233.604 | 52 | — |
| 224×224 / 1 / 50 | 50 | lerobot raw tuned | 2 | — | 817.345 | 1500.601 | 1567.200 | 1634.002 | 39 | — |
| 224×224 / 1 / 50 | 50 | zmq prefetch | 39 | — | 966.723 | 1666.794 | 1733.707 | 1733.905 | 38 | — |
| 224×224 / 1 / 50 | 50 | zmq sync | 36 | — | 866.694 | 1633.418 | 1699.848 | 1699.944 | 35 | — |
| 224×224 / 1 / 50 | 50 | **zmq async** | 2 | 0 | 600.335 | 1133.410 | 1166.795 | 1167.207 | 52 | 52.915 / 53.865 / 54.676 / 55.289 |
| 224×224 / 1 / 30 | 150 | lerobot jpeg default | 6 | — | 400.177 | 600.485 | 600.626 | 800.510 | 127 | — |
| 224×224 / 1 / 30 | 150 | lerobot jpeg tuned | 6 | — | 533.870 | 867.281 | 900.599 | 1136.174 | 77 | — |
| 224×224 / 1 / 30 | 150 | lerobot raw default | 5 | — | 400.155 | 600.502 | 600.616 | 767.131 | 127 | — |
| 224×224 / 1 / 30 | 150 | lerobot raw tuned | 5 | — | 533.827 | 867.275 | 900.589 | 1067.259 | 77 | — |
| 224×224 / 1 / 30 | 150 | zmq prefetch | 268 | — | 833.458 | 1233.486 | 1267.069 | 1267.252 | 66 | — |
| 224×224 / 1 / 30 | 150 | zmq sync | 212 | — | 666.849 | 1199.853 | 1233.262 | 1233.276 | 52 | — |
| 224×224 / 1 / 30 | 150 | **zmq async** | 5 | 0 | 367.184 | 600.036 | 600.199 | 600.506 | 128 | 152.937 / 153.739 / 154.282 / 154.686 |
| 224×224 / 1 / 30 | 50 | lerobot jpeg default | 3 | — | 300.161 | 500.464 | 500.608 | 600.535 | 128 | — |
| 224×224 / 1 / 30 | 50 | lerobot jpeg tuned | 3 | — | 500.225 | 867.271 | 900.599 | 1000.600 | 69 | — |
| 224×224 / 1 / 30 | 50 | lerobot raw default | 2 | — | 267.735 | 500.449 | 500.620 | 567.248 | 128 | — |
| 224×224 / 1 / 30 | 50 | lerobot raw tuned | 2 | — | 483.882 | 867.239 | 900.596 | 967.253 | 69 | — |
| 224×224 / 1 / 30 | 50 | zmq prefetch | 67 | — | 633.417 | 1033.477 | 1067.085 | 1067.446 | 66 | — |
| 224×224 / 1 / 30 | 50 | zmq sync | 59 | — | 533.421 | 1000.091 | 1033.251 | 1033.291 | 58 | — |
| 224×224 / 1 / 30 | 50 | **zmq async** | 2 | 0 | 267.006 | 500.024 | 500.242 | 500.599 | 128 | 52.873 / 53.595 / 54.967 / 55.695 |
| 224×224 / 1 / 16 | 150 | lerobot jpeg default | 5 | — | 267.108 | 333.889 | 333.968 | 369.834 | 371 | — |
| 224×224 / 1 / 16 | 150 | lerobot jpeg tuned | 6 | — | 300.509 | 433.868 | 433.964 | 633.780 | 198 | — |
| 224×224 / 1 / 16 | 150 | lerobot raw default | 5 | — | 266.841 | 333.782 | 333.930 | 336.057 | 379 | — |
| 224×224 / 1 / 16 | 150 | lerobot raw tuned | 5 | — | 300.487 | 433.855 | 433.977 | 600.550 | 198 | — |
| 224×224 / 1 / 16 | 150 | zmq prefetch | 556 | — | 600.112 | 800.306 | 800.404 | 800.548 | 137 | — |
| 224×224 / 1 / 16 | 150 | zmq sync | 360 | — | 433.574 | 733.271 | 766.599 | 766.605 | 89 | — |
| 224×224 / 1 / 16 | 150 | **zmq async** | 5 | 0 | 233.195 | 299.955 | 300.078 | 300.613 | 358 | 152.880 / 153.697 / 154.000 / 157.163 |
| 224×224 / 1 / 16 | 50 | lerobot jpeg default | 3 | — | 100.542 | 133.937 | 134.011 | 167.235 | 994 | — |
| 224×224 / 1 / 16 | 50 | lerobot jpeg tuned | 3 | — | 234.617 | 433.876 | 433.982 | 533.917 | 149 | — |
| 224×224 / 1 / 16 | 50 | lerobot raw default | 2 | — | 100.536 | 133.922 | 133.973 | 136.664 | 1034 | — |
| 224×224 / 1 / 16 | 50 | lerobot raw tuned | 2 | — | 235.393 | 433.859 | 433.948 | 500.608 | 149 | — |
| 224×224 / 1 / 16 | 50 | zmq prefetch | 139 | — | 400.132 | 600.309 | 600.404 | 600.470 | 138 | — |
| 224×224 / 1 / 16 | 50 | zmq sync | 106 | — | 300.098 | 566.510 | 566.598 | 566.609 | 105 | — |
| 224×224 / 1 / 16 | 50 | **zmq async** | 2 | 0 | 83.740 | 99.990 | 100.287 | 100.696 | 898 | 52.635 / 53.150 / 53.635 / 55.691 |

## Validation and limits

Full CPU suite: **656 passed, 22 subtests passed, 2 existing skips**, exit 0 ([log](cpu-suite.txt)). The two baseline skips are the original inapplicable jaw cases in `test_mimic_joints.py` and `test_parallel_gripper_geometry.py`; neither test was changed. No new skips, ignore flags or deselections were introduced. The only pre-existing test file modified is `test_transport.py`, which gained a real-ZMQ async example-server test; its existing assertions were retained. New worker/integration tests use controlled completions and a fake monotonic clock.

Default simulator evaluation remains step-driven: `SimulationSmoother`, fixed schedules, seeds, eval loop and simulator policy wrapper retain their existing default behavior. Temporal ensembling is explicitly rejected in async mode. Socket close drains the pending RPC sequence on the worker (each RPC retains its configured transport timeout); cancellation invalidates playback immediately and never waits on that socket.

Both documented client modes also passed against the real ZMQ example server on CPU ([utility validation](gpu-client-smoke.txt)); the GPU-serving command was checked through the CLI dry-run planner. This did not execute a GPU model or open a physical robot.

Hardware validation still required: real-arm clamp/first-target/Stop behavior under device latency and operator races, real GPU inference tail latency, calibrated action/units parity, trajectory jerk and physical smoothness. Fixed synthetic RGB/actions and one run per cell cannot establish these. The benchmark changes only the example plugin instance camera list for 1–3 cameras; production policy folders and transport are unchanged. Raw captures are kept locally under ignored `raw_async/`; committed files contain summaries and hashes.
