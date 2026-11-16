# Live demo script (2–3 minutes)

**Before the session** (5 minutes, on the laptop you will present from):

1. Plug in the charger; set the speaker volume to about 80%.
2. Turn **Audio enhancements off** for the microphone: Settings → System → Sound → Microphone Array → Audio enhancements: Off.
   Without this, Windows cancels steady machine noise and live-mic mode hears nothing.
3. Start the dashboard: `python demo/serve.py`, then open **http://localhost:8000** in Chrome and allow microphone access.
4. Open the Edge Impulse project (Impulse design and Model testing pages) in a second tab.
5. Put one normal and one abnormal pump clip on your phone, ready to play
   (`demo/clips/normal_1_id00.wav`, `demo/clips/abnormal_1_id00.wav`).
6. **Backup:** record a 1-minute screen video of a successful run (Win + Alt + R starts and stops recording).

## What to say and do

| Time | Say | Do |
|---|---|---|
| 0:00 | "Pumps fail gradually, and the sound changes before they break. We put a tiny neural network next to the machine that listens and raises an alarm, with no cloud needed." | Show the title slide |
| 0:20 | "We trained it in Edge Impulse on recordings of real factory pumps (DCASE 2020 / MIMII), plus our own laptop-microphone recordings. The impulse is a 1-second window → MFE features → 1D CNN, quantised to int8." | Edge Impulse tab: Impulse design |
| 0:45 | "On clips it never saw, it reaches **{test accuracy}**, quantised to int8. On an Arduino Nano 33 BLE Sense it would need about **{latency} ms** and **{RAM} KB** of RAM." | Model testing page |
| 1:05 | "Here it runs on this laptop as WebAssembly: same model, no internet." | Dashboard → Replay → a *normal* clip → shows NORMAL |
| 1:25 | "Now a faulty pump." | Replay an *abnormal* clip → shows ABNORMAL (red timeline) |
| 1:45 | "And live, through the microphone." | Live microphone → play the abnormal clip from the phone near the laptop |
| 2:05 | "Its limit: a sound it was never trained on is forced into a known class. That is why the next step is an anomaly block." | Replay the *white noise* clip → shows ABNORMAL |
| 2:20 | "The same model also compiles to C++ for microcontrollers. GitHub Actions retrains and rebuilds it every two days." | Show the C++ terminal output (report Section 5.2) and the Actions tab |

## If something goes wrong

| Problem | Fix |
|---|---|
| Live mode stays on BACKGROUND or does not react | Audio enhancements are still on, or the phone is too far away: hold it about 10 cm from the laptop's microphone (top edge of the screen). Switch to replay mode. |
| The page shows "Model missing" | Run `python scripts/ei_pipeline.py --skip-upload --skip-retrain --targets wasm`, then refresh. |
| The browser fails completely | Edge Impulse dashboard → *Launch in browser* (Edge Impulse's own live classification), or play the backup video. |
