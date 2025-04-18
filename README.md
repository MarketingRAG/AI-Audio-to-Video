# Audio to YouTube Shorts Converter

This project automates the process of converting audio content (like podcasts) into engaging YouTube Shorts by:
1. Transcribing the audio using OpenAI's Whisper
2. Creating a full-length video with subtitles
3. Using Google's Gemini AI to identify viral-worthy clips
4. Automatically extracting and saving the identified clips

## Features

- **Audio Transcription**: Converts audio files to text using OpenAI's Whisper
- **Video Generation**: Creates a full-length video with subtitles from audio and a background image
- **AI-Powered Clip Selection**: Uses Google's Gemini AI to identify potentially viral-worthy segments
- **Automatic Clip Extraction**: Extracts and saves the identified clips as separate video files
- **Subtitle Support**: Automatically generates and overlays subtitles on the video

## Prerequisites

- Python 3.7+
- FFmpeg
- Arial font installed (for subtitles)
- Google API key for Gemini AI

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/audio_to_youtube_shorts.git
cd audio_to_youtube_shorts
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Create an `audio_file` directory in the project root
2. Place your audio files (supported formats: .mp3, .wav, .m4a, .ogg) in the `audio_file` directory
3. Run the script:
   ```bash
   python audio_to_youtube_shorts.py
   ```

The script will:
- Process all audio files in the `audio_file` directory
- Create a separate output directory for each audio file
- Generate transcripts, full videos, and short clips for each file

Output will be organized as follows:
```
output/
  audio1/
    audio1_transcript.srt
    audio1_video.mp4
    audio1_clips.json
    clips/
      clip1.mp4
      clip2.mp4
      ...
  audio2/
    ...
```

## Output Files

- `full_transcript.srt`: The complete transcript in SRT format
- `output_video.mp4`: The full-length video with subtitles
- `selected_clips.json`: Information about the selected clips
- `clips/`: Directory containing the extracted short clips

## Requirements

The project uses the following Python packages:
- whisper
- moviepy
- pysrt
- numpy
- google-generativeai
- python-dotenv

## Notes

- The script uses the "base" model of Whisper for transcription
- Clips are selected based on criteria like emotional impact, surprising insights, and potential for visual representation
- The default clip duration is between 15-60 seconds
- Subtitles are automatically generated and styled with a semi-transparent black background

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 