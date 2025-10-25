# Audio Files for Manual Override System

## Required Sound Files

Place the following audio files in this directory:

1. **access_granted.mp3**
   - Sound for approved access
   - Recommended: Short beep or "Access Granted" voice
   - Duration: 1-2 seconds

2. **access_denied.mp3**
   - Sound for denied access
   - Recommended: Buzzer or "Access Denied" voice
   - Duration: 1-2 seconds

3. **manual_required.mp3**
   - Sound for manual review required
   - Recommended: Alert tone or chime
   - Duration: 1-2 seconds

4. **manual_override.mp3**
   - Sound for manual override actions
   - Recommended: Special tone or notification sound
   - Duration: 1-2 seconds

## Where to Get Sound Files

### Option 1: Free Sound Resources
- **Freesound.org**: https://freesound.org/
- **Zapsplat.com**: https://www.zapsplat.com/
- **Mixkit.co**: https://mixkit.co/free-sound-effects/

### Option 2: Text-to-Speech (TTS)
Generate voice files using:
- **Google TTS**: https://cloud.google.com/text-to-speech
- **Amazon Polly**: https://aws.amazon.com/polly/
- **Microsoft Azure TTS**: https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/

### Option 3: Create Simple Beeps with Audacity
1. Install Audacity (free audio editor)
2. Generate > Tone
3. For success: 880 Hz, 0.3s
4. For error: 440 Hz, 0.5s
5. Export as MP3

## Temporary Placeholder

If you don't have audio files yet, you can:
1. Create silent MP3 files (the system will work without sound)
2. Use placeholder files until you get proper audio

### Create Silent Placeholder (macOS/Linux):

```bash
# Install ffmpeg if needed
brew install ffmpeg  # macOS
# or
sudo apt install ffmpeg  # Ubuntu/Linux

# Create silent 1-second MP3 files
ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -q:a 9 -acodec libmp3lame access_granted.mp3
ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -q:a 9 -acodec libmp3lame access_denied.mp3
ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -q:a 9 -acodec libmp3lame manual_required.mp3
ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 1 -q:a 9 -acodec libmp3lame manual_override.mp3
```

## Audio Specifications

- **Format**: MP3
- **Bitrate**: 128 kbps (recommended)
- **Sample Rate**: 44.1 kHz
- **Channels**: Mono or Stereo
- **Max Duration**: 3 seconds
- **Volume**: Normalized to -3dB to prevent clipping

## Testing Audio

After adding audio files, test them in the browser console:

```javascript
const audio = new Audio('/static/sounds/access_granted.mp3');
audio.play();
```

## File Permissions

Ensure audio files are readable:

```bash
chmod 644 *.mp3
```

---

**Note**: The system will still work without audio files - alerts will be visual only until audio files are added.
