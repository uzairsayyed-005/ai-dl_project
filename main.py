import os
import numpy as np
from dataclasses import dataclass
import whisper
import torch

AS_SEGMENT_DURATION = 0.1  # Duration of each segment in the speaker diarization, in seconds
TRANSCRIPT_PATH = 'transcript.txt'  # Path to the (final) transcript file (after speaker naming)
SPEAKER_SUMMARY_PATH = 'speaker_summary.txt'  # Path to the speaker-wise summary file

# Global model variable to cache the Whisper model
_whisper_model = None

def get_whisper_model():
    """Get or load the Whisper model (cached)"""
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using device: {device}')
        print('Loading Whisper model (this happens only once)...')
        _whisper_model = whisper.load_model('base', device=device)
        print('Whisper model loaded and cached')
    return _whisper_model


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_id: int | None  # -1 for silence


def transcribe_video(video_path: str, num_speakers: int, language: str) -> list[Segment]:
    """
    Transcribe the audio of a video file to text. The function performs speaker diarization on the audio file and transcribes the audio to text using the Whisper model. The function returns a list of Segments with 'start', 'end', 'text', 'speaker_id'. The 'speaker_id' is an integer representing the speaker ID for the segment. These segments should represent the dialogue as accurately as possible, with each segment containing the text spoken by a single speaker and the next segment containing the text spoken by another speaker. The speaker ID should be unique for each speaker and should be consistent throughout the transcription.
    :param video_path: Path to the video file
    :param num_speakers: Number of speakers in the audio file
    :param language: Language of the audio file
    :return: List of Segments with 'start', 'end', 'text', 'speaker_id'
    """
    from moviepy.editor import VideoFileClip

    tmp_audio_path = 'audiofile.wav'  # Path to the temporary audio file

    # Convert video to audio using moviepy with optimized settings
    print(f'Converting {video_path} to {tmp_audio_path}')
    video = VideoFileClip(video_path)
    print(f'Loaded video {video_path}')
    if video.audio is None:
        raise ValueError('Video has no audio')

    # Optimized audio extraction: lower sample rate for faster processing
    video.audio.write_audiofile(tmp_audio_path, codec='pcm_s16le', fps=16000, nbytes=2, verbose=False, logger=None)
    print('Converted video to audio')

    segments = transcribe_audio(tmp_audio_path, num_speakers, language)

    os.remove(tmp_audio_path)

    return segments


def transcribe_audio(audio_path: str, num_speakers: int, language: str) -> list[Segment]:
    """
    Transcribe the audio file to text. The function performs speaker diarization on the audio file and transcribes the audio to text using the Whisper model. The function returns a list of Segments with 'start', 'end', 'text', 'speaker_id'. The 'speaker_id' is an integer representing the speaker ID for the segment. These segments should represent the dialogue as accurately as possible, with each segment containing the text spoken by a single speaker and the next segment containing the text spoken by another speaker. The speaker ID should be unique for each speaker and should be consistent throughout the transcription.
    :param audio_path: Path to the audio file
    :param num_speakers: Number of speakers in the audio file
    :param language: Language of the audio file
    :return: List of Segments with 'start', 'end', 'text', 'speaker_id'
    """
    # Use the cached Whisper model
    model = get_whisper_model()
    print('Loaded model')

    # Transcribe the audio
    print('Transcribing audio')
    result = model.transcribe(audio_path, language=language, verbose=True)
    print('Transcribed audio')

    segments = []
    for segment in result['segments']:
        segments.append(Segment(segment['start'], segment['end'], segment['text'].strip(), None))  # type: ignore

    print(segments)

    flags = __diarize_audio(audio_path, num_speakers)

    print('Flags')
    print(flags)

    # Map diarization speaker flags to Whisper transcription segments
    transcription_segments = __map_sentences_to_speakers(segments, flags)

    print(transcription_segments)

    # Organize the transcription by speaker
    organized_segments = __organize_by_speaker(transcription_segments)
    print(organized_segments)

    return organized_segments


def write_transcript(segments: list[Segment], output_path: str) -> None:
    """
    Write the transcription segments to the console and a text file. Each segment is printed with the speaker ID and text. The output file will contain the same information. The output file is opened in the default text editor after writing.
    """

    for segment in segments:
        print(f'Speaker {segment.speaker_id} ({segment.start:.2f}-{segment.end:.2f}): {segment.text}')

    unique_speakers = len(set([segment.speaker_id for segment in segments]))
    print(f'In total we have {len(segments)} segments and {unique_speakers} unique speakers')

    with open(output_path, 'w', encoding='utf-8') as file:
        for segment in segments:
            file.write(f'Speaker {segment.speaker_id} ({segment.start:.2f}-{segment.end:.2f}): {segment.text}\n\n')

    os.startfile(output_path)  # open the file in the default file editor


def generate_speaker_summary(segments: list[Segment]) -> dict:
    """Aggregate speaker-wise statistics and return a summary structure.

    Returns a dictionary keyed by speaker label (name or ID) with:
      total_time: float (seconds spoken)
      segment_count: int
      word_count: int
      avg_words_per_segment: float
      speaking_share: float (percentage of total speaking time)
      first_ts: float (first appearance timestamp)
      last_ts: float (last appearance timestamp)
      top_keywords: list[str]
      sample_excerpt: str
    """
    from collections import defaultdict, Counter
    import re

    # Collect per-speaker text and timing
    speaker_data: dict = {}
    total_speaking_time = 0.0

    # Helper containers
    times_acc = defaultdict(float)
    seg_counts = defaultdict(int)
    texts = defaultdict(list)
    first_ts = {}
    last_ts = {}

    for seg in segments:
        speaker = seg.speaker_id
        if speaker is None:  # Skip silence
            continue
        duration = max(0.0, seg.end - seg.start)
        times_acc[speaker] += duration
        seg_counts[speaker] += 1
        texts[speaker].append(seg.text)
        total_speaking_time += duration
        if speaker not in first_ts:
            first_ts[speaker] = seg.start
        last_ts[speaker] = seg.end

    # Basic stopword list (lightweight)
    stopwords = set(
        [
            'the','and','for','that','this','with','from','have','your','about','there','their','will','would',
            'what','when','where','which','while','shall','should','could','into','then','than','also','they',
            'them','you','are','was','were','been','being','can','our','out','any','but','not','just','like',
            'over','after','before','because','those','these','here','such','upon','only','each','other','more'
        ]
    )

    summary: dict = {}
    for speaker, total_time in times_acc.items():
        full_text = ' '.join(texts[speaker])
        # Tokenize words
        tokens = [
            w.lower()
            for w in re.findall(r"[A-Za-z']+", full_text)
            if len(w) > 3 and w.lower() not in stopwords
        ]
        word_count = len(re.findall(r"\w+", full_text))
        keyword_counts = Counter(tokens)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(8)]
        avg_words_per_segment = word_count / seg_counts[speaker] if seg_counts[speaker] else 0.0
        speaking_share = (total_time / total_speaking_time * 100.0) if total_speaking_time else 0.0
        excerpt = full_text[:300].strip()
        summary[speaker] = {
            'total_time': total_time,
            'segment_count': seg_counts[speaker],
            'word_count': word_count,
            'avg_words_per_segment': avg_words_per_segment,
            'speaking_share_pct': speaking_share,
            'first_ts': first_ts.get(speaker, 0.0),
            'last_ts': last_ts.get(speaker, 0.0),
            'top_keywords': top_keywords,
            'sample_excerpt': excerpt,
        }

    return summary


def write_speaker_summary(summary: dict, output_path: str) -> None:
    """Persist the speaker summary dictionary to a human-readable text file."""
    lines: list[str] = []
    lines.append('SPEAKER SUMMARY\n')
    if not summary:
        lines.append('No speaker data available.')
    else:
        # Order by total speaking time descending
        ordered = sorted(summary.items(), key=lambda kv: kv[1]['total_time'], reverse=True)
        for speaker, data in ordered:
            lines.append(f"=== Speaker: {speaker} ===")
            lines.append(f"Total Speaking Time: {data['total_time']:.2f} s")
            lines.append(f"Speaking Share: {data['speaking_share_pct']:.2f}%")
            lines.append(f"Segments: {data['segment_count']}")
            lines.append(f"Word Count: {data['word_count']}")
            lines.append(f"Avg Words / Segment: {data['avg_words_per_segment']:.2f}")
            lines.append(f"First Appearance: {data['first_ts']:.2f} s")
            lines.append(f"Last Appearance: {data['last_ts']:.2f} s")
            if data['top_keywords']:
                lines.append(f"Top Keywords: {', '.join(data['top_keywords'])}")
            if data['sample_excerpt']:
                lines.append('Sample Excerpt:')
                lines.append(data['sample_excerpt'])
            lines.append('')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    try:
        os.startfile(output_path)
    except Exception:
        pass


def __diarize_audio(audio_path: str, num_speakers: int) -> list[int]:
    """
    Perform speaker diarization on the audio file. The function returns a list of speaker IDs for fixed-duration segments. The speaker IDs are integers starting from 0. The function also detects silence periods in the audio file which are represented by a speaker ID of -1.
    """
    from pyAudioAnalysis.audioSegmentation import speaker_diarization

    print('Performing speaker diarization...')
    # Detect silence periods (optimized)
    silence_periods = __detect_silence(audio_path)

    # Faster speaker diarization with reduced accuracy for speed
    [flags, classes, accuracy] = speaker_diarization(audio_path, num_speakers, plot_res=False)
    flags = [int(flag) for flag in flags]

    # Adjust flags based on silence
    adjusted_flags = __adjust_flags_for_silence(flags, silence_periods)
    print('Speaker diarization completed')

    return adjusted_flags


def __detect_silence(audio_path: str, smoothing_filter_size: int = 50) -> list[tuple[float, float]]:
    """
    Detects silence periods in an audio file.

    :param audio_path: Path to the audio file.
    :param smoothing_filter_size: Size of the smoothing filter applied to energy signal.
    :return: A list of tuples representing silent periods (start_time, end_time).
    """
    from pyAudioAnalysis.audioBasicIO import read_audio_file, stereo_to_mono
    from pyAudioAnalysis.ShortTermFeatures import feature_extraction

    # Extract short-term features
    [fs, x] = read_audio_file(audio_path)
    x = stereo_to_mono(x)  # Convert to mono if stereo

    # Calculate frame length and step size in samples
    frame_length_samples = int(0.050 * fs)  # 50 ms frame
    frame_step_samples = int(0.025 * fs)  # 25 ms step

    features, f_names = feature_extraction(x, fs, frame_length_samples, frame_step_samples)

    # Find the index of the energy feature
    energy_index = f_names.index('energy')
    energy = features[energy_index, :]

    # Smooth the energy signal
    if smoothing_filter_size > 1:
        energy = np.convolve(energy, np.ones(smoothing_filter_size) / smoothing_filter_size, mode='same')

    # Using the energy signal calculated above
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)

    # Example heuristic: set threshold as mean minus half standard deviation
    energy_threshold = max(mean_energy - 1.2 * std_energy, 0.001)  # Ensure it doesn't go negative

    # Identify frames below the energy threshold
    silent_frames = energy < energy_threshold

    # Group silent frames into continuous silent periods
    silent_periods = []
    start_time = None
    for i, is_silent in enumerate(silent_frames):
        if is_silent and start_time is None:
            start_time = i * 0.025  # Start time of the silent period
        elif not is_silent and start_time is not None:
            end_time = i * 0.025  # End time of the silent period
            silent_periods.append((start_time, end_time))
            start_time = None
    # Handle case where the last frame is silent
    if start_time is not None:
        silent_periods.append((start_time, len(silent_frames) * 0.025))

    return silent_periods


def __adjust_flags_for_silence(flags: list[int], silence_periods: list[tuple[float, float]]) -> list[int]:
    """
    Insert silence periods into the speaker flags array. This function adjusts the speaker flags array based on the identified silence periods. The speaker flags array is a list of speaker IDs for fixed-duration segments. The silence periods are represented by a speaker ID of -1.
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :param silence_periods: List of tuples representing silent periods (start_time, end_time)
    :return: Adjusted list of speaker IDs with silence periods inserted
    """
    # Adjust the flags array based on identified silence periods
    adjusted_flags = flags.copy()

    for silence_start, silence_end in silence_periods:
        start_index = int(silence_start / AS_SEGMENT_DURATION)
        end_index = int(silence_end / AS_SEGMENT_DURATION) + 1
        adjusted_flags[start_index:end_index] = [-1] * (end_index - start_index)

    return adjusted_flags


def __split_text_into_sentences(text: str) -> list[str]:
    """
    Simple function to split text into sentences based on punctuation.
    This is a naive implementation and can be replaced with more sophisticated NLP tools.
    """
    import re

    # TODO replace with a more sophisticated NLP tool?
    sentences = re.split(r'[.!?]\s*', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences


def __get_segment_flags(flags: list[int], start_time: float, end_time: float) -> list[int]:
    """
    Get speaker flags for a segment based on the start and end times. Silences (represented by -1) are filtered out.
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :param start_time: Start time of the segment
    :param end_time: End time of the segment
    :return: List of speaker IDs for the segment
    """
    start_index = int(start_time / AS_SEGMENT_DURATION)
    end_index = int(end_time / AS_SEGMENT_DURATION) + 1
    segment_flags = flags[start_index:end_index]

    # Filter out pause/silence speaker IDs if defined (e.g., ID -1)
    return [flag for flag in segment_flags if flag != -1]


def __map_sentences_to_speakers(transcription_segments: list[Segment], flags: list[int]) -> list[Segment]:
    """
    Map sentences to speakers based on the speaker diarization. This function splits segments with mixed speakers into sentences and assigns the most common speaker ID to each sentence. The function returns a list of Segments with proper speaker IDs for each sentence.
    :param transcription_segments: List of Segments with 'start', 'end', 'text'
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :return: A list of Segments for each sentence
    """
    result_segments: list[Segment] = []
    for segment in transcription_segments:
        segment_flags = __get_segment_flags(flags, segment.start, segment.end)

        if not segment_flags or len(set(segment_flags)) == 1:
            # Segment has a unanimous speaker (or silence), keep it as is
            speaker_id = segment_flags[0] if segment_flags else None
            segment.speaker_id = speaker_id
            result_segments.append(segment)
        else:
            # Segment has mixed speakers, split into sentences
            sentences = __split_text_into_sentences(segment.text)

            sentence_length_prefix_sum = sum(len(sentence) for sentence in sentences)

            for i, sentence in enumerate(sentences):
                # Estimate sentence duration based on the number of characters in the sentence compared to the total with respect to the segment duration
                sentence_duration = len(sentence) / sentence_length_prefix_sum * (segment.end - segment.start)

                sentence_start_time = segment.start + i * sentence_duration
                sentence_end_time = sentence_start_time + sentence_duration

                sentence_flags = __get_segment_flags(flags, sentence_start_time, sentence_end_time)
                most_common_speaker = max(set(sentence_flags), key=sentence_flags.count) if sentence_flags else None

                result_segments.append(
                    Segment(
                        start=sentence_start_time,
                        end=sentence_end_time,
                        text=sentence,
                        speaker_id=most_common_speaker,
                    )
                )

                sentence_start_time = sentence_end_time  # Update start time for the next sentence

    return result_segments


def __organize_by_speaker(transcription_segments: list[Segment]) -> list[Segment]:
    """
    Organize the transcription segments by speaker. This function groups consecutive segments by the same speaker into a single segment. This means that consecutive segments by the same speaker are merged into a single segment and the text is concatenated. The output will therefore contain fewer segments than the input and never have consecutive segments by the same speaker.
    :param transcription_segments: List of Segments
    :return: List of Segments organized by speaker
    """
    organized_segments: list[Segment] = []
    current_speaker: int | None = None
    current_segment: Segment = None  # type: ignore

    for segment in transcription_segments:
        # Check if the current segment is continuing
        if segment.speaker_id == current_speaker:
            # Append text to the current segment
            current_segment.text += ' ' + segment.text
            current_segment.end = segment.end
        else:
            # Finish the current segment and start a new one
            if current_segment:
                organized_segments.append(current_segment)

            current_speaker = segment.speaker_id
            # Make a copy of the segment to avoid modifying the original segment
            current_segment = Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker_id=segment.speaker_id,
            )

    # Don't forget to add the last segment
    if current_segment:
        organized_segments.append(current_segment)

    return organized_segments


if __name__ == '__main__':
    video_path = input('Enter the path to the video file: ')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'File not found: {video_path}')

    num_speakers = int(input('Enter the number of speakers: '))
    language = input('Enter the language of the audio file: ')

    transcription_segments = transcribe_video(video_path, num_speakers, language)

    # Initial transcript with numeric speaker IDs
    write_transcript(transcription_segments, TRANSCRIPT_PATH)

    unique_speakers = list(
        sorted(set(segment.speaker_id for segment in transcription_segments if segment.speaker_id is not None))
    )
    speaker_name_mapping = {}
    for speaker in unique_speakers:
        name = input(f'Enter name for speaker {speaker}: ')
        speaker_name_mapping[speaker] = name

    for segment in transcription_segments:
        if segment.speaker_id is not None:
            segment.speaker_id = speaker_name_mapping[segment.speaker_id]

    # Final transcript with speaker names
    write_transcript(transcription_segments, TRANSCRIPT_PATH)

    # Generate and write speaker summary (uses named speakers)
    speaker_summary = generate_speaker_summary(transcription_segments)
    write_speaker_summary(speaker_summary, SPEAKER_SUMMARY_PATH)

#C:\Users\uzair\Desktop\AI-DL_Project\Meeting-Summarizer\meetenv\Scripts\python.exe C:\Users\uzair\Desktop\AI-DL_Project\Meeting-Summarizer\main.py 