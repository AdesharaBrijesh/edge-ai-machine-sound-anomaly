# Live demo script (2–3 minutes)

**Before the session** (5 minutes, on the laptop you will present from):

1. Plug in the charger; set the speaker volume to about 80%.
2. Turn **Audio enhancements off** for the microphone: Settings → System → Sound → Microphone Array → Audio enhancements: Off.
   Without this, Windows cancels steady machine noise and live-mic mode hears nothing.
3. Start the dashboard: `python demo/serve.py`, then open the address it prints (**http://localhost:8000**, or 8001 if 8000 is taken) in Chrome or Edge and allow microphone access.
4. Open the Edge Impulse project (Impulse design and Classifier pages) in a second browser tab. Set browser zoom to 100 % (Ctrl+0).
5. Put one normal and one abnormal pump clip on your phone, ready to play
   (`demo/clips/normal_1_id00.wav`, `demo/clips/abnormal_1_id00.wav`).
6. **Backup:** record a 1-minute screen video of a successful run (Win + Alt + R starts and stops recording).

## What to say and do

| Time | Say | Do |
|---|---|---|
| 0:00 | "Pumps fail gradually, and the sound changes before they break. We put a tiny neural network next to the machine that listens and raises an alarm, with no cloud." | Title slide |
| 0:20 | "It was trained in Edge Impulse on real factory pump recordings plus our own microphone recordings: 1-second windows, MFE features, a 1D CNN quantised to int8. This page runs that exact model in the browser." | Dashboard, **01 Monitor** |
| 0:40 | "A healthy pump." | Press **1** (normal clip 1): NORMAL, machine state HEALTHY |
| 0:55 | "Now a faulty one. Notice the alarm logic: one odd window never raises an alarm, six of the last eight do, and every state change is logged like a maintenance record." | Press **6** (abnormal clip 1): ABNORMAL, machine state ALARM, event log entry |
| 1:15 | "Live, through the microphone." | Press **M**, play the abnormal clip from the phone near the laptop; then stay quiet: NO MACHINE / BACKGROUND |
| 1:35 | "Which part of the sound does it use? We delete one frequency band at a time and re-run the model: 152 extra inferences." | **02 Analyze**, abnormal clip 1: the 300–600 Hz bar dominates |
| 1:55 | "Factories are loud. Here are all held-out clips with noise mixed in at seven levels, about 1,300 inferences in ten seconds." | **03 Stress test** → Run: within 5 points of clean down to 0 dB; errors only ever go to *abnormal* (false alarms, never missed faults) |
| 2:20 | "Everything here is the deployed model: 5,043 parameters, 7 ms on an Arduino, under a millisecond in the browser. The same library compiles to C++ for microcontrollers." | **04 Model card** → Run benchmark; mention the C++ output on the deploy slide |
| 2:40 | "Its limit: a sound it was never trained on is forced into a class; that's why the next step is an anomaly block." | Optional: Monitor → *white noise* clip shows ABNORMAL |

## If something goes wrong

| Problem | Fix |
|---|---|
| Live mode stays on BACKGROUND or does not react | Audio enhancements are still on, or the phone is too far away: hold it about 10 cm from the laptop's microphone (top edge of the screen). Switch to **Replay clip** in the 01 Monitor tab. |
| The page shows "Model missing" | Run `python scripts/ei_pipeline.py --skip-upload --skip-retrain --targets wasm`, then refresh. |
| The browser fails completely | Edge Impulse dashboard → *Launch in browser* (Edge Impulse's own live classification), or play the backup video. |
| Charts or the page look tiny/huge | Reset browser zoom with Ctrl+0, then Ctrl+F5. |
