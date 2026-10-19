# Phase 3 DL scheduler and TB handover validation

This directory records the focused high-load test used to validate RLC buffer
report routing and the fate of already-scheduled transport blocks during a
controlled physical-BWP switch.

## Reproduction command

```bash
./ns3 run "scratch/joint-energy-mvp \
  --enableTrueBwp=true --enableTrueBwpSwitchScript=true \
  --bwpSwitchToWideSec=0.4 --bwpSwitchBackToNarrowSec=0.7 \
  --bwpHandoverTraceCsv=/tmp/v6-dl-handover-tb-backlog.csv \
  --bwpHandoverSampleMs=0.25 \
  --tbContinuityTraceCsv=/tmp/v6-dl-handover-tb-events.csv \
  --controllerWarmupMs=100 --simTime=1 --numUes=3 --udpDataRate=40 \
  --enableGating=false --enableCdrx=false \
  --csvFileName=/tmp/v6-dl-handover-tb-summary.csv"
```

Numerology 1 gives a 0.5 ms slot. The scheduler probe sampled every 0.25 ms.
Traffic was constant downlink UDP at 40 Mbps per UE so both switches occurred
while data was queued.

## Scheduler-buffer result

| Switch | Target scheduler first non-zero backlog | Target BWP first DL symbols | Old BWP scheduled tail |
|---|---:|---:|---:|
| 20 -> 100 MHz | 0.25 ms | 1.75 ms | 4.25 ms |
| 100 -> 20 MHz | 0.25 ms | 1.75 ms | 1.25 ms |

The new scheduler receives an updated RLC buffer report within the first
0.25 ms sample after either switch. This is much shorter than the 40 ms policy
interval, so this run provides no justification for forcing an extra buffer
report or bypassing the scheduler to read RLC queues directly.

## TB continuity result

Across the full run, the UE PHY reported 5,349 received DL TBs and zero
corruptions. The gNB MAC processed 5,343 DL HARQ feedback records before the
simulation stop; every processed record was ACK and none was NACK.

After 20 -> 100 MHz, 27 TB receptions already in the BWP-0 pipeline completed
between 0.400178569 s and 0.404249997 s. After 100 -> 20 MHz, 9 BWP-1 TB
receptions completed between 0.700107141 s and 0.701249997 s. All were
uncorrupted. HARQ feedback whose arrival time fell after the switch was also
ACK-only.

The first target-BWP reception occurred 1.679 ms after either switch. During
20 -> 100 MHz, old- and new-BWP receptions overlap. During 100 -> 20 MHz, the
last old-BWP reception and first new-BWP reception are separated by about
0.429 ms, less than one slot.

## Interpretation and limitation

The current controlled switch changes the UE RRC primary DL/UL indices,
rebinds the target PHY/MAC RNTI, and changes the gNB per-UE BWP routing. It
does not stop the old UE PHY event loop. Consequently, already-scheduled TBs
on the old BWP remain receivable and drain successfully while scheduling
moves to the target BWP.

This proves continuity in the current simulator implementation, but it also
means the model does not yet enforce a strict single-active-DL-BWP receiver at
the switch boundary. Treat the successful old-pipeline drain as an explicit
v6 modeling choice/limitation, not as proof that old-BWP reception would be
available after an instantaneous RF retune in a standards-constrained UE.

## Files

- `backlog_trace.csv`: 0.25 ms scheduler backlog and per-BWP DL-symbol samples.
- `tb_events.csv`: per-BWP gNB TX, UE RX, corruption, and DL HARQ events.
- `run_summary.csv`: end-to-end QoS and official energy summary.
