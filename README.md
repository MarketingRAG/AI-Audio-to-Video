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
- FFmpeg: Must be installed and accessible in the system's PATH. Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).
- Font for Subtitles: The script uses a specific font for subtitles.
    - The default font path is `DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"` in `audio_to_youtube_shorts.py`. This is common on some Linux distributions.
    - Users on other operating systems (Windows, macOS) or Linux systems without this exact font path **must** update the `DEFAULT_FONT_PATH` variable in the script.
    - Set it to a valid .ttf or .otf font file on your system (e.g., "arial.ttf" if it's a system-wide font MoviePy can find, or a full path like "C:/Windows/Fonts/arial.ttf").
- Gemini API Key: You'll need a Gemini API key from Google.

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

3. Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Configuration

Several key parameters can be configured by editing the variables at the top of the `audio_to_youtube_shorts.py` script:

- `GEMINI_API_KEY`: Your Gemini API key from Google. This is typically set in the `.env` file (e.g., `GEMINI_API_KEY=your_api_key_here`) and loaded by the script.
- `DEFAULT_FONT_PATH`: The file path to the .ttf or .otf font file used for subtitles. Defaults to `"/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"`. Ensure this path is correct for your system, or update it.
- `DEFAULT_WHISPER_MODEL`: Controls the OpenAI Whisper model used for transcription. Options include "tiny", "base", "small", "medium", "large". Larger models offer higher accuracy but are slower and require more resources. Defaults to "base".
- `DEFAULT_BACKGROUND_IMAGE`: The filename of the default background image (e.g., "background.jpg") to be placed in the project's root directory. This image is used if the `BACKGROUND_IMAGES_DIR` is empty or if images within it are unusable.
- `BACKGROUND_IMAGES_DIR`: The name of the directory (e.g., "background_images") where you can place multiple images to be used as backgrounds for the videos. Images from this directory will be cycled through for different segments if available.
- `AUDIO_FILES_DIR`: The name of the directory (e.g., "audio_file") where the script looks for input audio files.
- `OUTPUT_DIR_BASE`: The name of the base directory (e.g., "output") where all processed files and subdirectories (one for each audio input) will be saved.

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
  {audio_basename}/
    {audio_basename}_transcript.srt
    {audio_basename}_video.mp4
    {audio_basename}_clips.json
    clips/
      {clip_title_sanitized}_{start_time_ms}.mp4
      {another_clip_title_sanitized}_{start_time_ms}.mp4
      ...
  {another_audio_basename}/
    ...
```

## Output Files

For each input audio file, a corresponding subdirectory is created in the `OUTPUT_DIR_BASE` (default is "output"). Let `{audio_basename}` be the name of your input audio file without the extension. The generated files include:

- `{audio_basename}_transcript.srt`: The complete transcript in SRT format.
- `{audio_basename}_video.mp4`: The full-length video with subtitles, using one of the background images.
- `{audio_basename}_clips.json`: A JSON file containing information about the selected clips, including timestamps and original text.
- `clips/` directory: Contains the extracted short video clips. Each clip is named using a sanitized version of its title and the start timestamp, e.g., `{clip_title_sanitized}_{start_time_ms}.mp4`.

## Dependencies

The project uses the following Python packages (see `requirements.txt` for exact versions):

- `google-generativeai>=0.3.0`
- `moviepy>=2.1.2`
- `python-dotenv>=1.0.0`
- `openai-whisper>=1.1.10` (Note: This package is installed as `openai-whisper` and imported in the script as `whisper`)
- `pysrt>=1.1.2`
- `numpy>=1.21.0`

## Notes

- The script uses the "base" model of Whisper for transcription
- Clips are selected based on criteria like emotional impact, surprising insights, and potential for visual representation
- The target clip duration is ideally 45-90 seconds, and clips must be at least 45 seconds long (as per the AI prompt).
- Subtitles are automatically generated and styled with a semi-transparent black background

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 