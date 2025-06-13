# Audio to YouTube Shorts Converter

This project automates the process of converting audio content (like podcasts) into engaging YouTube Shorts by:
1. Transcribing the audio using OpenAI's Whisper
2. Creating a full-length video with subtitles
3. Using Google's Gemini AI to identify viral-worthy clips
4. Automatically extracting and saving the identified clips
5. Providing a Streamlit UI for easy file uploads and processing.

## Features

- **Streamlit UI**: Easy-to-use interface for uploading files and generating clips.
- **Audio Transcription**: Converts audio files to text using OpenAI's Whisper
- **Video Generation**: Creates a full-length video with subtitles from audio and a background image
- **AI-Powered Clip Selection**: Uses Google's Gemini AI to identify potentially viral-worthy segments
- **Automatic Clip Extraction**: Extracts and saves the identified clips as separate video files
- **Subtitle Support**: Automatically generates and overlays subtitles on the video

## Prerequisites

- Python 3.7+
- FFmpeg: Must be installed and accessible in the system's PATH. Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).
- Font for Subtitles: The script `audio_to_youtube_shorts.py` uses a specific font for subtitles.
    - The default font path is `DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"`. This is common on some Linux distributions.
    - Users on other operating systems (Windows, macOS) or Linux systems without this exact font path **must** update the `DEFAULT_FONT_PATH` variable in `audio_to_youtube_shorts.py`.
    - Set it to a valid .ttf or .otf font file on your system (e.g., "arial.ttf" if it's a system-wide font MoviePy can find, or a full path like "C:/Windows/Fonts/arial.ttf").
- Gemini API Key: You'll need a Gemini API key from Google.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/audio_to_youtube_shorts.git
cd audio_to_youtube_shorts
```

2. Install the required Python packages (including Streamlit for the UI):
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Configuration

Several key parameters can be configured by editing the variables at the top of the `audio_to_youtube_shorts.py` script. These settings are used by the backend processing logic:

- `GEMINI_API_KEY`: Your Gemini API key from Google. This is typically set in the `.env` file (e.g., `GEMINI_API_KEY=your_api_key_here`) and loaded by the script.
- `DEFAULT_FONT_PATH`: The file path to the .ttf or .otf font file used for subtitles. Defaults to `"/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"`. Ensure this path is correct for your system, or update it.
- `DEFAULT_WHISPER_MODEL`: Controls the OpenAI Whisper model used for transcription. Options include "tiny", "base", "small", "medium", "large". Larger models offer higher accuracy but are slower and require more resources. Defaults to "base".
- `DEFAULT_BACKGROUND_IMAGE`: The filename of the default background image (e.g., "background.jpg") to be placed in the project's root directory or another accessible path. This image is used if the `BACKGROUND_IMAGES_DIR` is empty or if images within it are unusable.
- `BACKGROUND_IMAGES_DIR`: The name of the directory (e.g., "background_images") where you can place multiple images to be used as backgrounds for the videos. Images from this directory will be cycled through for different segments if available. This path is relative to where the script is run or should be an absolute path.
- `OUTPUT_DIR_BASE`: This constant is still present but primarily relevant if you adapt the `main()` function in `audio_to_youtube_shorts.py` for direct testing. The Streamlit app uses its own temporary directory structure.

## Running the Streamlit UI

The primary way to use this tool is via the Streamlit web interface.

1. Ensure you have installed the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (This includes Streamlit, which was recently added).

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open the URL provided by Streamlit in your web browser.
4. The UI will allow you to upload an audio or video file. After uploading, click "Generate Clips" to start processing.

## Output Files

When using the Streamlit UI, uploaded files and their outputs are stored in a temporary session directory (e.g., inside `/tmp/clip_generator_sessions/` on Linux). The application will display the exact output path for generated clips.

If you adapt `audio_to_youtube_shorts.py` for direct script execution, the output structure is as follows:
Let `output_base_dir` be the directory you specify and `{media_filename_base}` be the name of your input media file without the extension.
```
<output_base_dir>/
  {media_filename_base}/
    {media_filename_base}_transcript.srt
    {media_filename_base}_video.mp4
    {media_filename_base}_clips.json
    clips/
      {clip_title_sanitized}_{timestamp}.mp4
      ...
```

- `{media_filename_base}_transcript.srt`: The complete transcript in SRT format.
- `{media_filename_base}_video.mp4`: The full-length video with subtitles.
- `{media_filename_base}_clips.json`: A JSON file containing information about the selected clips.
- `clips/` directory: Contains the extracted short video clips.

## Dependencies

The project uses the following Python packages (see `requirements.txt` for exact versions):

- `google-generativeai>=0.3.0`
- `moviepy>=2.1.2`
- `python-dotenv>=1.0.0`
- `openai-whisper>=20240930`
- `pysrt>=1.1.2`
- `numpy>=1.21.0`
- `streamlit>=1.0.0` (New addition for the UI)

## Notes

- The script `audio_to_youtube_shorts.py` (used by the Streamlit app) uses the "base" model of Whisper for transcription by default.
- Clips are selected based on criteria like emotional impact, surprising insights, and potential for visual representation.
- The target clip duration is ideally 45-90 seconds, and clips must be at least 45 seconds long (as per the AI prompt).
- Subtitles are automatically generated and styled with a semi-transparent black background.
- Ensure FFmpeg is installed and accessible in your system PATH for video processing.
- Ensure the font file specified in `audio_to_youtube_shorts.py` (DEFAULT_FONT_PATH) exists and is accessible.

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.