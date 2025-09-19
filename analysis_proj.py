

# Meeting Summarizer: Summary Table Analysis Script
import os
import re
import wave

# === USER WORKFLOW ===
# 1. Run main.py as usual (e.g., python main.py)
# 2. If possible, keep the original audio file (e.g., audiofile.wav) in the folder.
# 3. (Optional, for full stats) Use the provided timing wrapper or add timing code to main.py to log processing time and RAM usage to 'processing_time.log'.
# 4. After main.py finishes, run this script to get a complete summary table.

TRANSCRIPT_PATH = 'transcript.txt'
SPEAKER_SUMMARY_PATH = 'speaker_summary.txt'
AUDIO_PATH = 'audiofile.wav'  # Change if your audio file is named differently
PROCESSING_LOG = 'processing_time.log'  # Log file for processing time and RAM

def get_audio_duration(audio_path=AUDIO_PATH):
    """Return audio duration in minutes if file exists, else None."""
    if not os.path.exists(audio_path):
        return None
    with wave.open(audio_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) / 60  # minutes

def analyze_transcript(transcript_path=TRANSCRIPT_PATH):
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    total_words = 0
    total_segments = 0
    speakers = set()
    for line in lines:
        match = re.match(r"Speaker (.+?) ", line)
        if match:
            speakers.add(match.group(1))
            total_segments += 1
            text = line.split(':', 1)[-1]
            total_words += len(re.findall(r'\w+', text))
    avg_words = total_words / total_segments if total_segments else 0
    return {
        'speakers': speakers,
        'segments': total_segments,
        'words': total_words,
        'avg_words': avg_words
    }

def analyze_speaker_summary(summary_path=SPEAKER_SUMMARY_PATH):
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()
    speakers = re.findall(r"=== Speaker: (.+?) ===", content)
    speaking_shares = [float(x) for x in re.findall(r"Speaking Share: ([\d.]+)%", content)]
    keywords = re.findall(r"Top Keywords: (.+)", content)
    return {
        'speakers': speakers,
        'speaking_shares': speaking_shares,
        'keywords': keywords
    }

def get_processing_time_and_ram(log_path=PROCESSING_LOG):
    """Parse processing_time.log for processing time (min) and peak RAM (GB)."""
    if not os.path.exists(log_path):
        return None, None
    with open(log_path, 'r') as f:
        lines = f.read()
    time_match = re.search(r"Processing time: ([\d.]+) seconds", lines)
    ram_match = re.search(r"Peak RAM: ([\d.]+) GB", lines)
    proc_time = float(time_match.group(1))/60 if time_match else None
    ram = float(ram_match.group(1)) if ram_match else None
    return proc_time, ram

def print_summary_table(audio_duration=None, proc_time=None, ram=None, tstats=None, sstats=None, wer=None, der=None):
    print("\nSummary Table\n" + "-"*40)
    print(f"{'Metric':<30} {'Value'}")
    print(f"{'-'*30} {'-'*20}")
    if audio_duration:
        print(f"{'Audio Duration':<30} {audio_duration:.2f} min")
    if proc_time:
        print(f"{'Processing Time':<30} {proc_time:.2f} min")
    if audio_duration and proc_time:
        print(f"{'Real-Time Factor (RTF)':<30} {proc_time/audio_duration:.2f}")
    if ram:
        print(f"{'Peak RAM Usage':<30} {ram:.2f} GB")
    if tstats:
        print(f"{'Speakers Detected':<30} {len(tstats['speakers'])}")
        print(f"{'Total Words Transcribed':<30} {tstats['words']}")
        print(f"{'Avg. Words per Segment':<30} {tstats['avg_words']:.2f}")
    if sstats:
        for i, speaker in enumerate(sstats['speakers']):
            print(f"{'Speaker ' + str(i+1) + ' Speaking Share':<30} {sstats['speaking_shares'][i]:.0f}%")
            print(f"{'Top Keywords (S' + str(i+1) + ')':<30} {sstats['keywords'][i] if i < len(sstats['keywords']) else ''}")
    if wer is not None:
        print(f"{'WER (if available)':<30} {wer*100:.1f}%")
    if der is not None:
        print(f"{'DER (if available)':<30} {der*100:.1f}%")
    print("-"*40)

# Optional: WER calculation if you have ground truth
def calculate_wer(reference, hypothesis):
    try:
        from jiwer import wer
        error = wer(reference, hypothesis)
        return error
    except ImportError:
        return None

if __name__ == "__main__":
    # Detect audio duration (minutes)
    audio_duration = get_audio_duration()  # None if file missing
    # Parse processing_time.log if present (see comments above for format)
    proc_time, ram = get_processing_time_and_ram()  # None if log missing
    # Analyze transcript and speaker summary
    tstats = analyze_transcript()
    sstats = analyze_speaker_summary()
    # For WER/DER, set these if you have reference data
    wer = None  # e.g., calculate_wer(reference_text, hypothesis_text)
    der = None  # set manually if available
    print_summary_table(audio_duration, proc_time, ram, tstats, sstats, wer, der)
