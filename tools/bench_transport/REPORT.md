# ZMQ versus LeRobot 0.7 remote inference

Status: INCOMPLETE — measurements/acceptance still in progress. 219/432 matrix cells recorded. See [audit](results/audit.json).

These CPU numbers isolate transport, processors and client scheduling. Neural inference is replaced by fixed zero chunks after 0/50/150 ms sleep. They do not establish GPU end-to-end latency, policy quality, physical smoothness or useful language behavior. [Exact commands and metric definitions](README.md).

## Measured facts

Software/hardware are recorded per run in `results/*/environment.json`. Client/server CPU percentages use process CPU time divided by wall time; one busy core is 100%. Request/response sizes below include captured loopback Ethernet/IP/TCP framing and ACKs, amortized over measured exchanges. This is packet accounting on `lo`, not physical link serialization. Application bytes and all four codec timing distributions are in [summary.csv](results/summary.csv); every RTT, codec sample, packet header/length and control tick is retained in compressed raw files.

### A — synchronous transport, zero inference delay

Each row has 50 warmups and 1,000 measured exchanges. Times are milliseconds. Sizes are bytes/exchange. C-enc/C-dec/S-enc/S-dec are mean codec times.

| H×W / cameras / chunk | Stack / encoding | p50 | p95 | p99 | max | C-enc | C-dec | S-enc | S-dec | Request B | Response B | Client CPU% | Server CPU% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 224×224 / 1 / 16 | lerobot / jpeg | 8.888 | 9.661 | 9.989 | 12.035 | 1.406 | 0.207 | 0.240 | 2.096 | 46309.784 | 1079.000 | 27.589 | 48.394 |
| 224×224 / 1 / 16 | lerobot / raw | 4.841 | 5.384 | 7.101 | 7.690 | 0.207 | 0.162 | 0.203 | 0.254 | 151510.784 | 1193.840 | 23.774 | 44.928 |
| 224×224 / 1 / 16 | zmq_npy / raw | 0.190 | 0.328 | 0.453 | 0.908 | 0.037 | 0.032 | 0.011 | 0.049 | 151244.132 | 745.538 | 63.708 | 49.006 |
| 224×224 / 1 / 16 | zmq_raw / raw | 0.065 | 0.307 | 0.873 | 1.771 | 0.014 | 0.003 | 0.002 | 0.010 | 151032.132 | 638.406 | 61.674 | 52.863 |
| 224×224 / 1 / 30 | lerobot / jpeg | 8.840 | 9.657 | 9.872 | 12.427 | 1.376 | 0.203 | 0.245 | 2.045 | 46309.784 | 1415.000 | 27.071 | 48.659 |
| 224×224 / 1 / 30 | lerobot / raw | 4.789 | 5.109 | 7.031 | 7.350 | 0.182 | 0.148 | 0.193 | 0.235 | 151510.850 | 1524.362 | 20.456 | 40.911 |
| 224×224 / 1 / 30 | zmq_npy / raw | 0.188 | 0.333 | 0.456 | 1.068 | 0.030 | 0.026 | 0.013 | 0.054 | 151244.132 | 1081.736 | 54.677 | 54.677 |
| 224×224 / 1 / 30 | zmq_raw / raw | 0.064 | 0.073 | 0.114 | 0.260 | 0.010 | 0.002 | 0.001 | 0.007 | 151032.132 | 974.934 | 57.204 | 71.505 |
| 224×224 / 1 / 50 | lerobot / jpeg | 8.800 | 9.586 | 9.855 | 11.604 | 1.267 | 0.187 | 0.214 | 1.853 | 46309.784 | 1895.000 | 26.593 | 46.877 |
| 224×224 / 1 / 50 | lerobot / raw | 4.856 | 5.544 | 7.035 | 7.488 | 0.219 | 0.175 | 0.201 | 0.258 | 151510.784 | 2004.494 | 25.505 | 45.989 |
| 224×224 / 1 / 50 | zmq_npy / raw | 0.200 | 0.262 | 0.627 | 1.894 | 0.032 | 0.030 | 0.014 | 0.059 | 151244.066 | 1561.208 | 49.862 | 58.928 |
| 224×224 / 1 / 50 | zmq_raw / raw | 0.065 | 0.145 | 0.229 | 0.627 | 0.010 | 0.002 | 0.001 | 0.008 | 151032.132 | 1453.812 | 61.074 | 48.860 |
| 224×224 / 2 / 16 | lerobot / jpeg | 9.686 | 11.866 | 13.647 | 16.094 | 1.827 | 0.153 | 0.194 | 2.734 | 92180.850 | 1206.446 | 29.320 | 53.044 |
| 224×224 / 2 / 16 | lerobot / raw | 4.938 | 6.905 | 7.228 | 8.042 | 0.279 | 0.180 | 0.193 | 0.311 | 302368.784 | 1259.774 | 29.691 | 55.502 |
| 224×224 / 2 / 16 | zmq_npy / raw | 0.252 | 0.501 | 0.651 | 0.756 | 0.058 | 0.033 | 0.016 | 0.093 | 302055.132 | 811.010 | 49.121 | 55.670 |
| 224×224 / 2 / 16 | zmq_raw / raw | 0.119 | 0.363 | 1.521 | 1.827 | 0.036 | 0.005 | 0.004 | 0.022 | 301739.132 | 704.208 | 60.857 | 44.260 |
| 224×224 / 2 / 30 | lerobot / jpeg | 5.568 | 10.310 | 10.942 | 12.943 | 0.962 | 0.074 | 0.092 | 1.528 | 92180.850 | 1542.116 | 21.746 | 40.431 |
| 224×224 / 2 / 30 | lerobot / raw | 4.838 | 5.225 | 7.013 | 7.292 | 0.223 | 0.143 | 0.179 | 0.280 | 302368.784 | 1602.044 | 23.966 | 50.978 |
| 224×224 / 2 / 30 | zmq_npy / raw | 0.255 | 0.577 | 0.953 | 1.649 | 0.061 | 0.034 | 0.016 | 0.096 | 302055.132 | 1146.680 | 55.301 | 58.373 |
| 224×224 / 2 / 30 | zmq_raw / raw | 0.098 | 0.336 | 0.410 | 0.773 | 0.022 | 0.003 | 0.002 | 0.017 | 301739.132 | 1039.680 | 66.256 | 58.894 |
| 224×224 / 2 / 50 | lerobot / jpeg | 6.126 | 10.847 | 11.188 | 13.471 | 1.252 | 0.103 | 0.129 | 2.076 | 92227.512 | 2026.538 | 24.531 | 47.155 |
| 224×224 / 2 / 50 | lerobot / raw | 4.935 | 7.005 | 7.308 | 8.877 | 0.273 | 0.176 | 0.194 | 0.312 | 302368.784 | 2079.206 | 28.935 | 55.150 |
| 224×224 / 2 / 50 | zmq_npy / raw | 0.258 | 0.856 | 1.524 | 2.059 | 0.071 | 0.040 | 0.019 | 0.115 | 302055.066 | 1626.944 | 53.682 | 59.050 |
| 224×224 / 2 / 50 | zmq_raw / raw | 0.133 | 0.343 | 0.682 | 1.772 | 0.028 | 0.003 | 0.002 | 0.018 | 301739.132 | 1518.096 | 64.220 | 46.706 |
| 224×224 / 3 / 16 | lerobot / jpeg | 5.438 | 6.458 | 6.651 | 12.403 | 0.810 | 0.045 | 0.063 | 1.334 | 138001.850 | 1145.000 | 18.852 | 37.170 |
| 224×224 / 3 / 16 | lerobot / raw | 4.760 | 7.103 | 9.337 | 9.615 | 0.222 | 0.116 | 0.153 | 0.282 | 453226.784 | 1384.712 | 22.978 | 51.351 |
| 224×224 / 3 / 16 | zmq_npy / raw | 0.281 | 0.417 | 1.373 | 2.448 | 0.070 | 0.031 | 0.014 | 0.102 | 452866.132 | 876.152 | 54.683 | 57.720 |
| 224×224 / 3 / 16 | zmq_raw / raw | 0.183 | 0.435 | 0.957 | 1.237 | 0.051 | 0.005 | 0.004 | 0.033 | 452446.132 | 770.670 | 65.659 | 48.150 |
| 224×224 / 3 / 30 | lerobot / jpeg | 5.722 | 14.237 | 15.760 | 17.331 | 1.361 | 0.077 | 0.101 | 2.148 | 138001.784 | 1481.000 | 24.637 | 45.754 |
| 224×224 / 3 / 30 | lerobot / raw | 4.965 | 7.253 | 7.892 | 9.456 | 0.294 | 0.153 | 0.188 | 0.352 | 453226.784 | 1707.116 | 28.010 | 61.319 |
| 224×224 / 3 / 30 | zmq_npy / raw | 0.314 | 1.431 | 1.810 | 2.189 | 0.092 | 0.042 | 0.020 | 0.148 | 452866.132 | 1213.208 | 52.567 | 59.138 |
| 224×224 / 3 / 30 | zmq_raw / raw | 0.157 | 0.288 | 1.313 | 1.894 | 0.045 | 0.004 | 0.003 | 0.025 | 452446.132 | 1105.350 | 65.462 | 50.355 |
| 224×224 / 3 / 50 | lerobot / jpeg | 10.405 | 16.774 | 17.420 | 20.141 | 2.308 | 0.140 | 0.167 | 3.519 | 138001.784 | 1961.000 | 30.150 | 53.505 |
| 224×224 / 3 / 50 | lerobot / raw | 5.098 | 7.285 | 7.657 | 9.742 | 0.331 | 0.174 | 0.193 | 0.376 | 453226.784 | 2199.854 | 31.431 | 61.752 |
| 224×224 / 3 / 50 | zmq_npy / raw | 0.314 | 0.429 | 1.176 | 1.967 | 0.071 | 0.031 | 0.014 | 0.104 | 452866.066 | 1692.680 | 52.404 | 58.227 |
| 224×224 / 3 / 50 | zmq_raw / raw | 0.159 | 0.322 | 0.468 | 0.695 | 0.040 | 0.003 | 0.003 | 0.027 | 452446.132 | 1585.218 | 64.589 | 53.824 |
| 480×640 / 1 / 16 | lerobot / jpeg | 10.115 | 15.487 | 22.429 | 25.806 | 2.267 | 0.076 | 0.089 | 3.621 | 277157.784 | 1211.528 | 29.182 | 54.002 |
| 480×640 / 1 / 16 | lerobot / raw | 9.093 | 10.389 | 11.530 | 12.355 | 0.474 | 0.149 | 0.182 | 0.427 | 923679.784 | 1650.164 | 28.669 | 58.285 |
| 480×640 / 1 / 16 | zmq_npy / raw | 0.911 | 1.360 | 2.161 | 2.873 | 0.543 | 0.036 | 0.017 | 0.144 | 923042.132 | 1073.228 | 77.688 | 29.259 |
| 480×640 / 1 / 16 | zmq_raw / raw | 0.285 | 0.595 | 1.010 | 2.871 | 0.080 | 0.004 | 0.003 | 0.068 | 922832.132 | 967.878 | 66.593 | 60.539 |
| 480×640 / 1 / 30 | lerobot / jpeg | 6.380 | 20.937 | 25.375 | 29.798 | 2.436 | 0.079 | 0.103 | 3.776 | 277157.784 | 1573.004 | 30.151 | 54.252 |
| 480×640 / 1 / 30 | lerobot / raw | 4.608 | 10.119 | 11.479 | 14.693 | 0.342 | 0.106 | 0.127 | 0.293 | 923679.850 | 1984.646 | 25.958 | 52.435 |
| 480×640 / 1 / 30 | zmq_npy / raw | 0.976 | 1.194 | 1.622 | 5.992 | 0.587 | 0.034 | 0.017 | 0.133 | 923042.132 | 1404.410 | 79.273 | 28.035 |
| 480×640 / 1 / 30 | zmq_raw / raw | 0.736 | 0.974 | 1.468 | 3.216 | 0.492 | 0.005 | 0.003 | 0.060 | 922832.132 | 1298.268 | 86.181 | 22.813 |
| 480×640 / 1 / 50 | lerobot / jpeg | 10.610 | 23.742 | 26.140 | 27.302 | 2.921 | 0.102 | 0.123 | 4.468 | 277157.784 | 2038.220 | 31.359 | 56.157 |
| 480×640 / 1 / 50 | lerobot / raw | 9.627 | 11.529 | 12.215 | 16.757 | 0.648 | 0.207 | 0.240 | 0.575 | 923679.784 | 2460.158 | 31.196 | 61.751 |
| 480×640 / 1 / 50 | zmq_npy / raw | 0.904 | 1.266 | 5.802 | 7.833 | 0.554 | 0.040 | 0.017 | 0.161 | 923042.066 | 1892.000 | 75.944 | 31.327 |
| 480×640 / 1 / 50 | zmq_raw / raw | 0.271 | 1.744 | 2.299 | 3.945 | 0.116 | 0.008 | 0.004 | 0.088 | 922832.132 | 1780.644 | 66.733 | 54.600 |
| 480×640 / 2 / 16 | lerobot / jpeg | 10.689 | 23.293 | 30.520 | 34.576 | 3.434 | 0.062 | 0.078 | 5.226 | 553594.784 | 1372.436 | 31.812 | 57.464 |
| 480×640 / 2 / 16 | lerobot / raw | 4.688 | 18.189 | 19.978 | 21.817 | 1.761 | 0.078 | 0.093 | 0.385 | 1846779.520 | 2254.592 | 43.418 | 51.493 |
| 480×640 / 2 / 16 | zmq_npy / raw | 1.314 | 1.616 | 2.099 | 7.815 | 0.311 | 0.035 | 0.018 | 0.370 | 1845783.132 | 1599.776 | 49.148 | 61.435 |
| 480×640 / 2 / 16 | zmq_raw / raw | 1.301 | 2.179 | 3.151 | 4.941 | 0.786 | 0.006 | 0.005 | 0.135 | 1845471.132 | 1467.102 | 82.950 | 28.822 |
| 480×640 / 2 / 30 | lerobot / jpeg | 10.586 | 14.757 | 21.181 | 28.825 | 2.863 | 0.050 | 0.067 | 4.458 | 553594.916 | 1721.768 | 30.717 | 56.841 |
| 480×640 / 2 / 30 | lerobot / raw | 6.611 | 18.686 | 19.706 | 22.550 | 2.080 | 0.092 | 0.113 | 0.456 | 1846779.850 | 2588.546 | 43.748 | 52.113 |
| 480×640 / 2 / 30 | zmq_npy / raw | 0.931 | 1.829 | 5.461 | 5.911 | 0.332 | 0.040 | 0.020 | 0.280 | 1845783.132 | 1926.338 | 62.774 | 51.280 |
| 480×640 / 2 / 30 | zmq_raw / raw | 1.494 | 2.091 | 2.914 | 7.117 | 0.748 | 0.005 | 0.004 | 0.245 | 1845471.132 | 1810.230 | 68.742 | 40.362 |
| 480×640 / 2 / 50 | lerobot / jpeg | 10.850 | 15.210 | 22.242 | 30.242 | 2.960 | 0.053 | 0.065 | 4.744 | 553594.784 | 2218.994 | 30.652 | 58.005 |
| 480×640 / 2 / 50 | lerobot / raw | 4.576 | 16.387 | 19.222 | 22.055 | 1.483 | 0.065 | 0.076 | 0.329 | 1846779.586 | 3105.572 | 43.077 | 51.025 |
| 480×640 / 2 / 50 | zmq_npy / raw | 0.827 | 1.754 | 3.124 | 8.006 | 0.304 | 0.037 | 0.019 | 0.260 | 1845783.066 | 2405.546 | 62.557 | 51.635 |
| 480×640 / 2 / 50 | zmq_raw / raw | 1.177 | 1.760 | 2.415 | 8.839 | 0.594 | 0.005 | 0.005 | 0.164 | 1845471.132 | 2307.126 | 75.507 | 36.509 |
| 480×640 / 3 / 16 | lerobot / jpeg | 14.800 | 21.910 | 27.071 | 37.014 | 4.397 | 0.060 | 0.076 | 6.627 | 830109.784 | 1509.584 | 31.695 | 57.527 |
| 480×640 / 3 / 16 | lerobot / raw | 6.844 | 12.046 | 17.053 | 19.826 | 0.587 | 0.061 | 0.077 | 0.512 | 2769880.586 | 2879.084 | 29.377 | 63.461 |
| 480×640 / 3 / 16 | zmq_npy / raw | 1.211 | 2.633 | 4.379 | 7.466 | 0.425 | 0.041 | 0.021 | 0.384 | 2768458.066 | 2046.398 | 61.827 | 52.796 |
| 480×640 / 3 / 16 | zmq_raw / raw | 0.900 | 4.544 | 5.840 | 6.997 | 0.356 | 0.008 | 0.007 | 0.249 | 2768044.132 | 1943.160 | 66.172 | 54.002 |
| 480×640 / 3 / 30 | lerobot / jpeg | 14.692 | 21.916 | 30.352 | 39.162 | 4.213 | 0.060 | 0.074 | 6.516 | 830109.784 | 1847.036 | 31.786 | 58.433 |
| 480×640 / 3 / 30 | lerobot / raw | 6.876 | 12.231 | 17.419 | 22.311 | 0.626 | 0.066 | 0.084 | 0.534 | 2769880.586 | 3224.918 | 28.934 | 64.504 |
| 480×640 / 3 / 30 | zmq_npy / raw | 1.219 | 2.198 | 3.738 | 5.172 | 0.411 | 0.040 | 0.020 | 0.368 | 2768458.132 | 2385.500 | 62.632 | 52.436 |
| 480×640 / 3 / 30 | zmq_raw / raw | 1.008 | 4.439 | 5.168 | 6.420 | 0.419 | 0.010 | 0.007 | 0.280 | 2768044.132 | 2278.830 | 67.755 | 52.913 |
| 480×640 / 3 / 50 | lerobot / jpeg | 16.743 | 26.484 | 33.096 | 40.003 | 4.940 | 0.064 | 0.082 | 7.418 | 830109.784 | 2214.176 | 32.136 | 57.026 |
| 480×640 / 3 / 50 | lerobot / raw | 7.372 | 17.296 | 19.782 | 22.753 | 0.760 | 0.083 | 0.099 | 0.644 | 2769880.916 | 3678.716 | 30.347 | 67.731 |
| 480×640 / 3 / 50 | zmq_npy / raw | 1.207 | 2.610 | 4.329 | 6.811 | 0.424 | 0.041 | 0.021 | 0.387 | 2768458.066 | 2863.454 | 61.903 | 54.165 |
| 480×640 / 3 / 50 | zmq_raw / raw | 0.852 | 1.088 | 1.467 | 2.736 | 0.256 | 0.005 | 0.003 | 0.166 | 2768044.132 | 2757.774 | 68.014 | 52.404 |

### B — 30 Hz control, 60 seconds per cell

Held includes startup, exhausted-buffer holds and deadlines missed during blocking calls. Missed is the blocking subset. Age is elapsed client monotonic time since the source observation, including the retained last target on holds; startup holds without a source are excluded. Blended actions use the older contributing observation. Replacements count accepted chunks after the first within the 60 s window, even if an early replacement supersedes an unexecuted chunk. A final blocking call can return after the window; its arrival is retained separately and excluded from replacement counts.

Our synchronous driver and prefetch setup call the real production `RemotePolicy.next_raw_action`. `SimulationSmoother` still queries synchronously and schedules synthetic P-step delivery; prefetch does not create a background network worker. Prefetch P=ceil(delay×30)+2 (4 or 7), blend=2. LeRobot uses its actual engine/dispatch with aligned chunks, zero blending, default refill 0.5 s or tuned delay+2/30 s (0.1167/0.2167 s). Its effective refill has an adaptive turnaround+one-action floor. This is a wall-clock paced control schedule with a retained last target; the unpaced Isaac simulator loop has different wall-clock behavior.

| H×W / cameras / chunk | Stack / encoding / setup | Delay ms | Held / 1800 | Missed | Age p50 ms | Age p95 ms | Age p99 ms | Age max ms | Replacements |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 480×640 / 1 / 16 | zmq_npy / raw / sync | 150.000 | 360.000 | 360.000 | 433.560 | 733.284 | 766.591 | 766.626 | 89.000 |
| 480×640 / 1 / 16 | zmq_npy / raw / prefetch | 50.000 | 139.000 | 139.000 | 400.126 | 600.313 | 600.423 | 600.856 | 138.000 |
| 480×640 / 1 / 16 | zmq_npy / raw / sync | 50.000 | 106.000 | 106.000 | 300.087 | 566.512 | 566.598 | 566.634 | 105.000 |

### C — added loopback latency

`tc netem` runs only inside the dedicated namespace. A configured delay of 1 or 5 ms applies on each traversal of `lo`, so a request/reply adds approximately 2 or 10 ms RTT before segmentation/ACK effects. Packet offloads remain at their defaults. This tests added delay, not a bandwidth-constrained WAN. All 72 A cells are repeated at each setting.

| H×W / cameras / chunk | Stack / encoding | Added delay per direction ms | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---:|---:|---:|---:|---:|
| 224×224 / 1 / 16 | lerobot / jpeg | 1.000 | 11.190 | 12.923 | 13.342 | 14.110 |
| 224×224 / 1 / 16 | lerobot / raw | 1.000 | 7.061 | 8.569 | 9.292 | 10.386 |
| 224×224 / 1 / 16 | zmq_npy / raw | 1.000 | 3.210 | 3.909 | 4.171 | 4.761 |
| 224×224 / 1 / 16 | zmq_raw / raw | 1.000 | 2.582 | 2.980 | 3.145 | 5.776 |
| 224×224 / 1 / 30 | lerobot / jpeg | 1.000 | 11.024 | 11.922 | 13.097 | 13.891 |
| 224×224 / 1 / 30 | lerobot / raw | 1.000 | 6.933 | 8.709 | 9.156 | 9.982 |
| 224×224 / 1 / 30 | zmq_npy / raw | 1.000 | 3.299 | 3.815 | 3.996 | 4.570 |
| 224×224 / 1 / 30 | zmq_raw / raw | 1.000 | 2.588 | 2.961 | 3.123 | 3.605 |
| 224×224 / 1 / 50 | lerobot / jpeg | 1.000 | 11.007 | 11.944 | 13.148 | 13.839 |
| 224×224 / 1 / 50 | lerobot / raw | 1.000 | 6.948 | 8.637 | 9.312 | 11.062 |
| 224×224 / 1 / 50 | zmq_npy / raw | 1.000 | 3.227 | 3.982 | 4.236 | 4.695 |
| 224×224 / 1 / 50 | zmq_raw / raw | 1.000 | 2.691 | 3.055 | 3.187 | 3.444 |
| 224×224 / 2 / 16 | lerobot / jpeg | 1.000 | 12.892 | 13.786 | 16.887 | 17.287 |
| 224×224 / 2 / 16 | lerobot / raw | 1.000 | 7.047 | 7.790 | 9.036 | 10.261 |
| 224×224 / 2 / 16 | zmq_npy / raw | 1.000 | 3.663 | 4.378 | 4.721 | 5.356 |
| 224×224 / 2 / 16 | zmq_raw / raw | 1.000 | 2.757 | 3.231 | 3.387 | 3.591 |
| 224×224 / 2 / 30 | lerobot / jpeg | 1.000 | 12.994 | 14.903 | 16.662 | 18.878 |
| 224×224 / 2 / 30 | lerobot / raw | 1.000 | 7.061 | 8.987 | 9.245 | 9.743 |
| 224×224 / 2 / 30 | zmq_npy / raw | 1.000 | 3.469 | 4.260 | 4.609 | 5.050 |
| 224×224 / 2 / 30 | zmq_raw / raw | 1.000 | 2.767 | 3.170 | 3.310 | 3.709 |
| 224×224 / 2 / 50 | lerobot / jpeg | 1.000 | 8.121 | 13.646 | 14.887 | 17.256 |
| 224×224 / 2 / 50 | lerobot / raw | 1.000 | 7.074 | 8.842 | 9.334 | 9.720 |
| 224×224 / 2 / 50 | zmq_npy / raw | 1.000 | 3.534 | 4.149 | 4.531 | 4.921 |
| 224×224 / 2 / 50 | zmq_raw / raw | 1.000 | 2.687 | 3.093 | 3.323 | 3.929 |
| 224×224 / 3 / 16 | lerobot / jpeg | 1.000 | 8.373 | 18.744 | 19.050 | 20.172 |
| 224×224 / 3 / 16 | lerobot / raw | 1.000 | 7.312 | 9.548 | 11.069 | 12.206 |
| 224×224 / 3 / 16 | zmq_npy / raw | 1.000 | 3.952 | 4.506 | 5.060 | 5.537 |
| 224×224 / 3 / 16 | zmq_raw / raw | 1.000 | 2.949 | 3.472 | 3.813 | 4.054 |
| 224×224 / 3 / 30 | lerobot / jpeg | 1.000 | 14.307 | 19.065 | 19.832 | 22.833 |
| 224×224 / 3 / 30 | lerobot / raw | 1.000 | 7.219 | 9.037 | 9.716 | 11.302 |
| 224×224 / 3 / 30 | zmq_npy / raw | 1.000 | 3.825 | 4.511 | 5.034 | 5.556 |
| 224×224 / 3 / 30 | zmq_raw / raw | 1.000 | 2.964 | 3.396 | 3.614 | 3.805 |
| 224×224 / 3 / 50 | lerobot / jpeg | 1.000 | 16.862 | 19.475 | 20.658 | 23.427 |
| 224×224 / 3 / 50 | lerobot / raw | 1.000 | 7.129 | 9.278 | 9.779 | 11.929 |
| 224×224 / 3 / 50 | zmq_npy / raw | 1.000 | 3.910 | 4.568 | 4.993 | 5.362 |
| 224×224 / 3 / 50 | zmq_raw / raw | 1.000 | 2.880 | 3.281 | 3.601 | 4.441 |
| 480×640 / 1 / 16 | lerobot / jpeg | 1.000 | 14.013 | 26.305 | 27.682 | 32.269 |
| 480×640 / 1 / 16 | lerobot / raw | 1.000 | 11.334 | 12.411 | 13.455 | 14.503 |
| 480×640 / 1 / 16 | zmq_npy / raw | 1.000 | 4.905 | 7.213 | 7.643 | 7.987 |
| 480×640 / 1 / 16 | zmq_raw / raw | 1.000 | 3.570 | 4.672 | 5.506 | 6.433 |
| 480×640 / 1 / 30 | lerobot / jpeg | 1.000 | 18.480 | 27.375 | 28.306 | 32.965 |
| 480×640 / 1 / 30 | lerobot / raw | 1.000 | 11.529 | 12.501 | 13.694 | 14.500 |
| 480×640 / 1 / 30 | zmq_npy / raw | 1.000 | 4.202 | 6.915 | 7.363 | 7.697 |
| 480×640 / 1 / 30 | zmq_raw / raw | 1.000 | 5.377 | 6.635 | 6.943 | 7.873 |
| 480×640 / 1 / 50 | lerobot / jpeg | 1.000 | 17.521 | 26.043 | 27.844 | 29.032 |
| 480×640 / 1 / 50 | lerobot / raw | 1.000 | 11.451 | 12.441 | 13.734 | 15.030 |
| 480×640 / 1 / 50 | zmq_npy / raw | 1.000 | 4.877 | 8.158 | 9.736 | 10.865 |
| 480×640 / 1 / 50 | zmq_raw / raw | 1.000 | 3.576 | 4.349 | 4.871 | 6.059 |
| 480×640 / 2 / 16 | lerobot / jpeg | 1.000 | 17.672 | 32.940 | 36.258 | 41.929 |
| 480×640 / 2 / 16 | lerobot / raw | 1.000 | 8.832 | 19.259 | 21.314 | 23.717 |
| 480×640 / 2 / 16 | zmq_npy / raw | 1.000 | 5.219 | 9.617 | 10.437 | 12.353 |
| 480×640 / 2 / 16 | zmq_raw / raw | 1.000 | 5.123 | 8.397 | 9.088 | 11.930 |
| 480×640 / 2 / 30 | lerobot / jpeg | 1.000 | 18.353 | 31.478 | 37.387 | 44.270 |
| 480×640 / 2 / 30 | lerobot / raw | 1.000 | 8.897 | 18.941 | 20.673 | 22.556 |
| 480×640 / 2 / 30 | zmq_npy / raw | 1.000 | 6.351 | 7.819 | 10.126 | 11.278 |
| 480×640 / 2 / 30 | zmq_raw / raw | 1.000 | 5.035 | 10.935 | 11.585 | 14.108 |
| 480×640 / 2 / 50 | lerobot / jpeg | 1.000 | 12.860 | 24.338 | 34.457 | 40.332 |
| 480×640 / 2 / 50 | lerobot / raw | 1.000 | 10.992 | 19.538 | 21.256 | 25.762 |
| 480×640 / 2 / 50 | zmq_npy / raw | 1.000 | 6.373 | 7.524 | 8.974 | 10.264 |
| 480×640 / 2 / 50 | zmq_raw / raw | 1.000 | 5.133 | 9.775 | 11.552 | 13.276 |
| 480×640 / 3 / 16 | lerobot / jpeg | 1.000 | 20.153 | 29.353 | 35.863 | 44.982 |
| 480×640 / 3 / 16 | lerobot / raw | 1.000 | 11.118 | 18.542 | 21.767 | 24.823 |
| 480×640 / 3 / 16 | zmq_npy / raw | 1.000 | 5.222 | 8.750 | 9.883 | 11.075 |
| 480×640 / 3 / 16 | zmq_raw / raw | 1.000 | 6.258 | 7.741 | 8.573 | 10.220 |
| 480×640 / 3 / 30 | lerobot / jpeg | 1.000 | 18.533 | 28.972 | 35.309 | 40.596 |
| 480×640 / 3 / 30 | lerobot / raw | 1.000 | 11.394 | 20.116 | 22.479 | 24.908 |
| 480×640 / 3 / 30 | zmq_npy / raw | 1.000 | 5.618 | 8.754 | 9.264 | 10.246 |
| 480×640 / 3 / 30 | zmq_raw / raw | 1.000 | 5.084 | 6.803 | 7.891 | 11.057 |
| 480×640 / 3 / 50 | lerobot / jpeg | 1.000 | 20.199 | 31.426 | 37.358 | 49.230 |
| 480×640 / 3 / 50 | lerobot / raw | 1.000 | 11.262 | 19.618 | 23.159 | 25.784 |
| 480×640 / 3 / 50 | zmq_npy / raw | 1.000 | 3.504 | 5.375 | 8.277 | 9.300 |
| 480×640 / 3 / 50 | zmq_raw / raw | 1.000 | 5.163 | 6.729 | 7.963 | 9.355 |
| 224×224 / 1 / 16 | lerobot / jpeg | 5.000 | 19.433 | 20.380 | 21.426 | 22.765 |
| 224×224 / 1 / 16 | lerobot / raw | 5.000 | 15.449 | 16.636 | 17.436 | 17.747 |
| 224×224 / 1 / 16 | zmq_npy / raw | 5.000 | 11.621 | 12.420 | 12.750 | 12.880 |
| 224×224 / 1 / 16 | zmq_raw / raw | 5.000 | 10.761 | 11.281 | 11.564 | 11.727 |
| 224×224 / 1 / 30 | lerobot / jpeg | 5.000 | 19.465 | 20.673 | 21.722 | 22.205 |
| 224×224 / 1 / 30 | lerobot / raw | 5.000 | 15.475 | 16.677 | 17.602 | 17.990 |
| 224×224 / 1 / 30 | zmq_npy / raw | 5.000 | 11.592 | 12.371 | 12.715 | 13.307 |
| 224×224 / 1 / 30 | zmq_raw / raw | 5.000 | 10.831 | 11.578 | 11.796 | 11.908 |
| 224×224 / 1 / 50 | lerobot / jpeg | 5.000 | 19.442 | 20.120 | 20.713 | 22.503 |
| 224×224 / 1 / 50 | lerobot / raw | 5.000 | 15.443 | 16.729 | 17.615 | 17.974 |
| 224×224 / 1 / 50 | zmq_npy / raw | 5.000 | 11.760 | 12.502 | 12.770 | 13.208 |
| 224×224 / 1 / 50 | zmq_raw / raw | 5.000 | 10.865 | 11.393 | 11.684 | 11.925 |
| 224×224 / 2 / 16 | lerobot / jpeg | 5.000 | 21.632 | 22.943 | 24.067 | 26.345 |
| 224×224 / 2 / 16 | lerobot / raw | 5.000 | 15.542 | 17.047 | 17.908 | 18.682 |
| 224×224 / 2 / 16 | zmq_npy / raw | 5.000 | 12.025 | 12.756 | 13.129 | 13.982 |
| 224×224 / 2 / 16 | zmq_raw / raw | 5.000 | 10.990 | 11.539 | 11.783 | 12.230 |
| 224×224 / 2 / 30 | lerobot / jpeg | 5.000 | 21.517 | 22.891 | 24.723 | 26.221 |
| 224×224 / 2 / 30 | lerobot / raw | 5.000 | 15.635 | 17.243 | 17.894 | 20.073 |
| 224×224 / 2 / 30 | zmq_npy / raw | 5.000 | 11.994 | 12.785 | 13.289 | 13.801 |
| 224×224 / 2 / 30 | zmq_raw / raw | 5.000 | 10.908 | 11.551 | 11.922 | 12.115 |
| 224×224 / 2 / 50 | lerobot / jpeg | 5.000 | 21.703 | 23.127 | 25.175 | 26.675 |
| 224×224 / 2 / 50 | lerobot / raw | 5.000 | 15.586 | 16.766 | 17.879 | 19.250 |
| 224×224 / 2 / 50 | zmq_npy / raw | 5.000 | 12.024 | 12.868 | 13.286 | 13.769 |
| 224×224 / 2 / 50 | zmq_raw / raw | 5.000 | 11.020 | 11.410 | 11.527 | 11.926 |
| 224×224 / 3 / 16 | lerobot / jpeg | 5.000 | 27.120 | 28.479 | 29.194 | 31.217 |
| 224×224 / 3 / 16 | lerobot / raw | 5.000 | 17.031 | 19.521 | 20.169 | 20.714 |
| 224×224 / 3 / 16 | zmq_npy / raw | 5.000 | 12.344 | 13.188 | 13.767 | 14.067 |
| 224×224 / 3 / 16 | zmq_raw / raw | 5.000 | 11.166 | 11.748 | 12.080 | 12.603 |
| 224×224 / 3 / 30 | lerobot / jpeg | 5.000 | 27.159 | 28.827 | 29.764 | 32.039 |
| 224×224 / 3 / 30 | lerobot / raw | 5.000 | 15.969 | 18.322 | 19.940 | 20.269 |
| 224×224 / 3 / 30 | zmq_npy / raw | 5.000 | 12.341 | 13.191 | 13.546 | 14.352 |
| 224×224 / 3 / 30 | zmq_raw / raw | 5.000 | 11.142 | 11.623 | 11.982 | 12.061 |
| 224×224 / 3 / 50 | lerobot / jpeg | 5.000 | 26.972 | 28.562 | 29.747 | 31.635 |
| 224×224 / 3 / 50 | lerobot / raw | 5.000 | 16.686 | 18.721 | 20.116 | 22.147 |
| 224×224 / 3 / 50 | zmq_npy / raw | 5.000 | 12.306 | 13.238 | 13.702 | 14.405 |
| 224×224 / 3 / 50 | zmq_raw / raw | 5.000 | 11.183 | 11.870 | 12.264 | 12.858 |
| 480×640 / 1 / 16 | lerobot / jpeg | 5.000 | 34.156 | 36.143 | 37.322 | 42.177 |
| 480×640 / 1 / 16 | lerobot / raw | 5.000 | 19.866 | 20.636 | 21.964 | 24.328 |
| 480×640 / 1 / 16 | zmq_npy / raw | 5.000 | 15.105 | 15.853 | 16.393 | 17.655 |
| 480×640 / 1 / 16 | zmq_raw / raw | 5.000 | 11.577 | 12.423 | 12.951 | 13.476 |
| 480×640 / 1 / 30 | lerobot / jpeg | 5.000 | 34.283 | 36.364 | 37.405 | 40.766 |
| 480×640 / 1 / 30 | lerobot / raw | 5.000 | 19.887 | 20.709 | 21.891 | 24.439 |
| 480×640 / 1 / 30 | zmq_npy / raw | 5.000 | 15.331 | 16.269 | 16.794 | 18.035 |
| 480×640 / 1 / 30 | zmq_raw / raw | 5.000 | 14.201 | 14.969 | 15.573 | 16.378 |
| 480×640 / 1 / 50 | lerobot / jpeg | 5.000 | 33.884 | 36.188 | 36.863 | 40.544 |
| 480×640 / 1 / 50 | lerobot / raw | 5.000 | 19.932 | 20.767 | 21.942 | 23.726 |
| 480×640 / 1 / 50 | zmq_npy / raw | 5.000 | 15.419 | 17.210 | 17.749 | 18.901 |
| 480×640 / 1 / 50 | zmq_raw / raw | 5.000 | 11.758 | 12.514 | 12.998 | 13.412 |
| 480×640 / 2 / 16 | lerobot / jpeg | 5.000 | 40.592 | 49.698 | 53.500 | 58.102 |
| 480×640 / 2 / 16 | lerobot / raw | 5.000 | 27.211 | 29.492 | 31.221 | 34.064 |
| 480×640 / 2 / 16 | zmq_npy / raw | 5.000 | 14.856 | 18.797 | 19.200 | 20.281 |
| 480×640 / 2 / 16 | zmq_raw / raw | 5.000 | 16.410 | 17.597 | 18.184 | 20.226 |
| 480×640 / 2 / 30 | lerobot / jpeg | 5.000 | 39.658 | 48.584 | 51.541 | 55.518 |
| 480×640 / 2 / 30 | lerobot / raw | 5.000 | 27.250 | 29.553 | 31.060 | 33.196 |
| 480×640 / 2 / 30 | zmq_npy / raw | 5.000 | 14.698 | 18.730 | 19.210 | 20.073 |
| 480×640 / 2 / 30 | zmq_raw / raw | 5.000 | 17.480 | 19.773 | 20.338 | 21.689 |
| 480×640 / 2 / 50 | lerobot / jpeg | 5.000 | 33.258 | 48.345 | 52.689 | 54.811 |
| 480×640 / 2 / 50 | lerobot / raw | 5.000 | 27.104 | 29.365 | 31.075 | 32.078 |
| 480×640 / 2 / 50 | zmq_npy / raw | 5.000 | 14.747 | 16.405 | 18.176 | 20.739 |
| 480×640 / 2 / 50 | zmq_raw / raw | 5.000 | 16.279 | 17.496 | 18.426 | 19.445 |
| 480×640 / 3 / 16 | lerobot / jpeg | 5.000 | 43.912 | 53.093 | 59.035 | 64.183 |
| 480×640 / 3 / 16 | lerobot / raw | 5.000 | 28.052 | 30.797 | 32.275 | 40.011 |
| 480×640 / 3 / 16 | zmq_npy / raw | 5.000 | 16.555 | 17.326 | 18.432 | 20.842 |
| 480×640 / 3 / 16 | zmq_raw / raw | 5.000 | 14.531 | 15.813 | 19.930 | 21.475 |
| 480×640 / 3 / 30 | lerobot / jpeg | 5.000 | 44.581 | 54.819 | 59.746 | 65.413 |
| 480×640 / 3 / 30 | lerobot / raw | 5.000 | 27.600 | 30.394 | 32.118 | 35.058 |
| 480×640 / 3 / 30 | zmq_npy / raw | 5.000 | 16.548 | 17.286 | 18.737 | 19.640 |
| 480×640 / 3 / 30 | zmq_raw / raw | 5.000 | 14.493 | 15.212 | 15.890 | 17.111 |
| 480×640 / 3 / 50 | lerobot / jpeg | 5.000 | 43.780 | 54.620 | 59.568 | 63.942 |
| 480×640 / 3 / 50 | lerobot / raw | 5.000 | 27.927 | 31.086 | 32.682 | 36.984 |
| 480×640 / 3 / 50 | zmq_npy / raw | 5.000 | 16.503 | 17.131 | 17.752 | 21.615 |
| 480×640 / 3 / 50 | zmq_raw / raw | 5.000 | 14.487 | 15.214 | 16.391 | 20.095 |

### Cold-session delivery and warmup

The initial 5 ms / three-camera 480×640 / LeRobot raw attempt timed out during warmup and stopped the supervisor. A fresh diagnostic reproduced a timeout on request 0: the client encoded one observation and the server decoded none. In a separate six-request diagnostic using one session, request 0 timed out at about 5,010 ms and the next five independent requests completed in 27–61 ms. Netem and the packet sniffer both reported zero drops. Captured client TCP payload, including admission, was smaller than the six encoded observation payloads. These observations are consistent with the transport's documented nonblocking DROP publication under startup backpressure; the exact internal Zenoh drop mechanism was not instrumented.

Original failures, counters and the diagnostic packet lengths are retained in [failures/](results/failures/), including [six-request evidence](results/failures/netem5-cold-session-delivery.json). The benchmark now records up to three warmup timeouts while still requiring 50 successful warmups before sampling. Each subsequent observation has a new request ID; no request is replayed. The 1,000 measured requests remain fail-fast, so successful steady-state rows do not discard failed measured samples. This diagnostic/manual-client warmup behavior is not evidence that RemoteInferenceEngine automatically recovers from such a startup timeout.

| Network | H×W / cameras / chunk | Stack / encoding | Successful warmups | Attempts | Warmup timeouts |
|---|---|---|---:|---:|---:|
| netem-5ms | 480×640 / 3 / 16 | lerobot / raw | 50 | 51 | 1 |
| netem-5ms | 480×640 / 3 / 30 | lerobot / raw | 50 | 51 | 1 |
| netem-5ms | 480×640 / 3 / 50 | lerobot / raw | 50 | 51 | 1 |

## Compatibility

Compatibility findings from source inspection (inferences, not model acceptance)

1. **The existing GR00T N1.7 and MolmoAct2 artifacts cannot be passed directly to `lerobot-policy-server`.** Its loader resolves a LeRobot `PreTrainedConfig`, loads a registered policy class, restores saved pre/postprocessor pipelines, validates the chunk contract and warms it. Our [GR00T wrapper](../../policies/groot_n17/server.py) uses NVIDIA `Gr00tPolicy.get_action` with split arm/gripper state, embodiment tags and offline Cosmos support. Our [MolmoAct2 wrapper](../../policies/molmoact2/server.py) loads a Transformers remote-code artifact and calls `predict_action` with its processor, `norm_tag`, camera ordering and explicit precision corrections. Neither is a LeRobot export with saved pipelines. [Loader source](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/remote_inference/deployment.py).

   LeRobot 0.7 does contain both policy families. Family support does not prove equivalence for our fine-tuned weights or processors. Two migration paths are plausible: convert/export each checkpoint into its LeRobot family with saved processors and prove parity, or register a `PreTrainedConfig`/`PreTrainedPolicy` adapter around our existing model wrapper. An adapter must implement `predict_action_chunk`, `reset`, `chunk_inference_spec` (single-current-observation chunks, prediction and execution horizons), canonical feature validation and saved preprocessing/postprocessing. Preserve GR00T embodiment/offline assets, joint splitting and action concatenation; preserve MolmoAct2 normalization, old coordinate convention, camera order and FP32 action/denormalization behavior. Verify repeated predictions, reset/seeds, action units and numerical parity on recorded observations before robot acceptance. Separate server environments can preserve incompatible model dependencies. LeRobot's native MolmoAct2 path can initialize original HF weights via `checkpoint_path`, but that evaluation path does not bypass the remote deployment loader's saved-artifact requirements. [GR00T integration](https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/groot.mdx), [MolmoAct2 original versus LeRobot artifacts](https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/molmoact2.mdx), [base policy API](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py).

2. **Map ordered state/action names explicitly and retain the richer local codec.** Our [`ArmInfo`](../../src/core/policy_client/contract.py) contains ordered joint names, arm/gripper roles, revolute/prismatic types and per-joint physical/device endpoints. Map `feature_names` to both state and action `FeatureSpec.names`, six-dimensional float32 shapes here, camera HWC uint8 schemas plus view-specific semantics, and negotiate 1/30 s action intervals. LeRobot admission compares feature fields/order, action feature, semantic convention, interval, execution mode and artifact identity. Its `semantics` strings identify a convention; they do not perform calibration or unit conversion. Encode a version/hash of joint types, roles, units, device mode, direction and calibration in a verified deployment convention, while retaining the actual mapping in the adapter. Do not relabel encoder-centred body degrees, gripper percent, radians or prismatic metres as interchangeable merely because shapes match. A changed calibration needs a new verified contract. [Admission implementation](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/remote_inference/client.py), [feature contract](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/inference/contracts.py).

3. **A simulator loop can drive the public library API.** The upstream loopback test admits a `RemoteClient`, constructs `RemoteInferenceEngine`, submits canonical observations with capture times and calls `dispatch_tick` with caller-owned `send` and `hold` callbacks, without a rollout robot object. This benchmark exercises that same path for every LeRobot B cell. An integration into [`src/sim/eval/loop.py`](../../src/sim/eval/loop.py) would capture images/state in the checkpoint's convention, send mapped targets into `env.step`, retain the last applied target on holds, clear episode state and close the worker on teardown. Our current loop advances simulation in fixed steps and can run faster/slower than wall time; LeRobot uses wall-clock monotonic freshness/worker scheduling, so either pace the integration in real time or explicitly validate a simulation clock adaptation. Driver hold checks belong to `lerobot-rollout` setup; supplying an engine directly leaves truthful hold behavior to the caller. This benchmark does not exercise Isaac dynamics. [Public library test](https://github.com/huggingface/lerobot/blob/v0.7.0/tests/remote_inference/test_library_client.py), [dispatch implementation](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/inference/dispatch.py).

4. **CUDA 13 alone is not a new conflict on this workstation.** Measured [`sim-environment.json`](results/sim-environment.json) reports the running Isaac Lab 3.0 derived container using Torch `2.11.0+cu130`, CUDA `13.0`, LeRobot `0.6.1`. [`gpu-driver.txt`](results/gpu-driver.txt) records driver `580.173.02`, above the release's stated `580.65` minimum. The repository [Dockerfile](../../docker/sim/Dockerfile) starts from `nvcr.io/nvidia/isaac-lab:3.0.0`, overrides Torch `2.11.0`/Torchvision `0.26.0`, NumPy `2.5.1` and Packaging `26.0`. Replacing that environment with an unconstrained 0.7 install is still a dependency/ABI risk: 0.7 constrains NumPy below 2.3, and model extras change Transformers and other packages. Keep the simulator untouched and run 0.7 serving in a separate environment/container first. This CPU benchmark uses `2.11.0+cpu`; it does not validate any CUDA kernels or model-specific precision patches. [Release installation changes](https://github.com/huggingface/lerobot/releases/tag/v0.7.0), [dependency constraints](https://github.com/huggingface/lerobot/blob/v0.7.0/pyproject.toml).

5. **Manual `/subtask` updates can work after the action adapter; `/vqa` needs a text backend.** `/subtask <text>` calls `RolloutController.set_task` and then `RemoteInferenceEngine.set_task`. It changes the instruction used for subsequent action inference, versions the intent and handles pending old-task work; it does **not** require `supports_text_generation` or `language.enabled`. Our GR00T and MolmoAct2 wrappers already consume an instruction per request. They still need the action export/adapter in question 1 to participate in LeRobot remote inference. Following a changed instruction usefully remains checkpoint dependent. [Command handlers](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/rollout/interactive.py), [controller](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/rollout/controller.py), [engine task changes](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/remote_inference/engine.py).

   `/vqa` asks for an answer; `/autosteer` asks for generated next instructions. Through the policy server's native language path, these need a policy declaring `supports_text_generation`, a `generate_text` implementation, compatible saved language processors/prompts and `language.enabled=true`. Our current wrappers expose action prediction rather than that API, and the native GR00T/MolmoAct2 classes in this release do not override the base text capability. Native server language work uses the same serialized model worker, requests a local hold and resumes action inference from fresh observations. [Text capability defaults](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/policies/pretrained.py), [native language execution](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/inference/policy_runner.py).

   A separate supported route is `--planner.model_id=<VLM>` with the `annotations` extra and an OpenAI-compatible VLM endpoint. This supplies an external text backend independently of the action policy's text head and works with the remote backend. Its VQA runs on a separate worker while eligible action execution continues. Starting external `/autosteer` deliberately pauses motion until the first accepted instruction; subsequent instruction changes follow the engine's task-transition rules. Supply `--planner.instructions` explicitly for remote policies because their saved recipe/training vocabulary is not available on the client. Upstream documents hardware exercise of this planner with local inference only, so remote/model/physical acceptance is still required. `/subtask` remains a direct manual update and cancels automatic planning. This CPU benchmark does not run a VLM or establish language quality. [External planner guide](https://github.com/huggingface/lerobot/blob/v0.7.0/docs/source/inference.mdx#external-planner), [external query routing](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/inference/base.py), [planner implementation](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/rollout/planner.py).


## Recommendation and migration work

Recommendation is pending the complete A/B matrix. No final migration decision is claimed from smoke tests or partially collected cells.

## Acceptance and limits

One run per cell, seeded synthetic high-entropy RGB, localhost TCP, no CPU affinity or hard real-time guarantees. JPEG q90 reduces bytes at a lossy encoding/decoding cost; real camera compressibility may differ. The observer adds lightweight timing instrumentation and packet capture overhead. Fixed actions measure continuity/freshness, not trajectory jerk or policy responsiveness. Tiny ACT is loaded/warmed with saved processors, then only its chunk prediction entry point is replaced; processor and protocol costs remain real.

The full CPU suite log is [cpu-tests-final.txt](results/cpu-tests-final.txt), with its [process exit code](results/cpu-tests-final.exitcode). The approved fixture correction retains both import embodiments and applies the two parallel-specific checks only to their matching fixture. Production invariance, raw-data checks, matrix coverage and test evidence are audited by `report.py --require-complete`. The GPU follow-up commands in README are intentionally not part of the CPU acceptance run.

Outstanding acceptance items:

- control: 3/216 cells, 213 missing, 0 extra
