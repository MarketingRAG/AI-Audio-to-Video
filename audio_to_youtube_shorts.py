# Step 1: Import the necessary libraries
import logging
import whisper
import os
from datetime import timedelta
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip, ColorClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips
import pysrt
import numpy as np
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv
import json
import re
from datetime import datetime

# Step 2: Configure the LLM API
# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# --- Configuration ---
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf" 
# On some systems, this might be: "Arial" if the font is installed system-wide and MoviePy can find it.
# For more robust cross-platform behavior, providing a path or ensuring font is in a known location is better.
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_BACKGROUND_IMAGE = "background.jpg" # Used if no images are in BACKGROUND_IMAGES_DIR
BACKGROUND_IMAGES_DIR = "background_images"
AUDIO_FILES_DIR = "audio_file"
OUTPUT_DIR_BASE = "output"
# --- End Configuration ---

# Step 3: Define utility functions for video and subtitle processing
def format_timedelta(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def create_srt(segments, output_file):
    """Create SRT file from Whisper segments"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_timedelta(segment['start'])
            end_time = format_timedelta(segment['end'])
            text = segment['text'].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

def clean_json_response(text):
    """Clean and fix the JSON response from Gemini."""
    text = text.replace('```json', '').replace('```', '')
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r'}\s*{', '},{', text)
    return text.strip()

def analyze_transcript(transcript):
    """Send transcript to Gemini API and get viral-worthy clip timestamps."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Step 3: Writing the prompt
    prompt = f"""
    You are an expert at identifying viral-worthy content from podcast transcripts. Your task is to analyze the following transcript and identify 3-5 clips that would perform well on social media platforms.

    Instructions:
    1. For each clip, provide:
       - Start and end timestamps in SRT format (HH:MM:SS,mmm)
       - A catchy title for the clip. The title should be short, relevant to the content and inspire people to watch the clip.
       - The exact text content of the clip
    2. Focus on clips that:
       - Have strong emotional impact
       - Contain surprising or counterintuitive insights
       - Include clear, concise statements
       - Have potential for visual representation
       - Are self-contained (45-90 seconds ideal)
       - Must be at least 45 seconds
    3. Ensure each clip:
       - Has a clear beginning and end
       - Makes sense without additional context
       - Is a complete thought or idea

    Return ONLY a valid JSON object with the following structure:
    {{
        "clips": [
            {{
                "start_time": "00:01:30,000",
                "end_time": "00:02:30,000",
                "title": "90% of marketers are doing this wrong",
                "content": "The actual transcript text here"
            }}
        ]
    }}

    Transcript to analyze:
    {transcript}
    """
    
    # Shorten transcript for logging to avoid overly long messages
    logged_transcript = transcript[:500] + "..." if len(transcript) > 500 else transcript
    logging.info(f"Sending transcript to Gemini API for analysis (first 500 chars): {logged_transcript}")

    try:
        response = model.generate_content(prompt)
        logging.info("Successfully received response from Gemini API.")
        return response.text
    except google_exceptions.InvalidArgument as e:
        logging.error(f"Invalid argument provided to Gemini API: {str(e)}. Please check the prompt and transcript data.")
        return None
    except google_exceptions.PermissionDenied as e:
        logging.error(f"Permission denied for Gemini API: {str(e)}. Please ensure your API key is correct and has the necessary permissions.")
        return None
    except google_exceptions.ResourceExhausted as e:
        logging.error(f"Gemini API resource exhausted: {str(e)}. You may have exceeded your API quota. Please check your quota and usage.")
        return None
    except google_exceptions.ServiceUnavailable as e:
        logging.error(f"Gemini API service unavailable: {str(e)}. The service may be temporarily down. Please try again later.")
        return None
    except google_exceptions.GoogleAPICallError as e: # More generic Google API error
        logging.error(f"A Google API call error occurred with Gemini: {str(e)}")
        return None
    except Exception as e: # Catch any other exception
        logging.exception("An unexpected error occurred during Gemini content generation:")
        return None

def parse_timestamp(timestamp_str):
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds. Returns None on parsing error."""
    try:
        if not isinstance(timestamp_str, str):
            # This case should ideally be caught by prior validation, but good to have.
            logging.error(f"Invalid type for timestamp: expected str, got {type(timestamp_str)} for value '{timestamp_str}'")
            raise ValueError("Timestamp must be a string.")
        
        parts = timestamp_str.split(':')
        if len(parts) != 3:
            raise ValueError("Timestamp must be in HH:MM:SS,mmm format.")
        
        hours = int(parts[0])
        minutes = int(parts[1])
        
        sec_millisec = parts[2].split(',')
        if len(sec_millisec) != 2:
            raise ValueError("Timestamp seconds and milliseconds part must be separated by a comma.")
            
        seconds = int(sec_millisec[0])
        milliseconds = int(sec_millisec[1])
        
        if not (0 <= hours <= 99 and 0 <= minutes <= 59 and 0 <= seconds <= 59 and 0 <= milliseconds <= 999):
            raise ValueError("Timestamp component out of valid range.")
            
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    except ValueError as e:
        logging.error(f"Error parsing timestamp string '{timestamp_str}': {str(e)}")
        return None
    except Exception as e: 
        logging.exception(f"An unexpected error occurred parsing timestamp '{timestamp_str}':")
        return None

def generate_subtitle_clip(text, start_time, duration, video_size):
    """Generate a subtitle clip with the given text and timing."""
    if not os.path.exists(DEFAULT_FONT_PATH):
        logging.warning(f"Font file '{DEFAULT_FONT_PATH}' not found. TextClip generation might fail or use a MoviePy default font if available.")
        # MoviePy will error if a specific font path is given and it's not found. 
        # If just a name like "Arial" is given, MoviePy might find it.

    txt_clip = TextClip(
        text=text,
        size=(int(video_size[0] * 0.85), int(video_size[1] * 0.2)),
        color='white',  # Changed from '#C8102E' to white for better visibility
        font=DEFAULT_FONT_PATH,
        stroke_color='black',  # Changed from '#FAFAFA' to black for better contrast
        stroke_width=3,  # Reduced from 4 to 3 for cleaner look
        method='caption'
    )
    txt_clip = txt_clip.with_duration(duration).with_start(start_time)
    txt_clip = txt_clip.with_position(('center', 'center'))
    return txt_clip

def get_background_images(directory=BACKGROUND_IMAGES_DIR):
    """Get all image files from the specified directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = []
    
    if not os.path.exists(directory):
        logging.info(f"Directory {directory} does not exist. Creating it...")
        try:
            os.makedirs(directory)
            logging.info(f"Successfully created directory: {directory}")
            return [] 
        except OSError as e:
            logging.error(f"Could not create directory {directory}: {str(e)}. Please check permissions or create the directory manually.")
            return [] 
    
    if not os.access(directory, os.R_OK | os.X_OK): 
        logging.error(f"Directory {directory} is not readable or accessible.")
        return []

    try:
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                    image_files.append(file_path)
                elif not os.path.isfile(file_path):
                    logging.warning(f"Found potential image '{file_path}' but it is not a file. Skipping.")
                elif not os.access(file_path, os.R_OK):
                    logging.warning(f"Found image file '{file_path}' but it is not readable. Skipping.")
    except OSError as e:
        logging.error(f"Error accessing contents of directory {directory}: {str(e)}")
        return []
    
    return image_files

# --- Helper functions for create_full_video ---

def _prepare_background_segments(subtitles, available_background_images, determined_video_size, audio_duration, default_fallback_image_path):
    """
    Prepares a list of ImageClip or ColorClip segments for the video background.
    Cycles through available_background_images. Uses a black ColorClip as fallback.
    Prepares a list of ImageClip or ColorClip segments for the video background.
    These clips will only have their duration set, not start times.
    Cycles through available_background_images. Uses a black ColorClip as fallback.
    Returns a list of video segments for concatenation and the (potentially updated) video_size.
    """
    video_segments_for_concatenation = []
    video_size = determined_video_size  # May be None initially
    current_image_index = -1

    # Case 1: No subtitles
    if not subtitles:
        segment_duration = audio_duration
        clip_to_add = None
        image_path_for_segment = None
        source_type = "ColorClip" # Default for logging if no image is used

        try:
            if available_background_images:
                image_path_for_segment = available_background_images[0]
                logging.info(f"No subtitles. Using first available background image: '{image_path_for_segment}' for full duration.")
                temp_image_clip = ImageClip(image_path_for_segment)
                if video_size is None:
                    video_size = temp_image_clip.size
                    logging.info(f"Video size determined from '{image_path_for_segment}': {video_size}")
                if temp_image_clip.size != video_size:
                    resized_clip = temp_image_clip.resized(video_size)
                    temp_image_clip.close()
                    temp_image_clip = resized_clip
                clip_to_add = temp_image_clip.with_duration(segment_duration)
                source_type = f"ImageClip ('{image_path_for_segment}')"
            elif default_fallback_image_path and os.path.isfile(default_fallback_image_path) and os.access(default_fallback_image_path, os.R_OK):
                image_path_for_segment = default_fallback_image_path
                logging.info(f"No subtitles and no images in {BACKGROUND_IMAGES_DIR}. Using default fallback image: '{image_path_for_segment}' for full duration.")
                temp_image_clip = ImageClip(image_path_for_segment)
                if video_size is None:
                    video_size = temp_image_clip.size
                    logging.info(f"Video size determined from '{image_path_for_segment}': {video_size}")
                if temp_image_clip.size != video_size:
                    resized_clip = temp_image_clip.resized(video_size)
                    temp_image_clip.close()
                    temp_image_clip = resized_clip
                clip_to_add = temp_image_clip.with_duration(segment_duration)
                source_type = f"ImageClip ('{image_path_for_segment}')"
            else:
                if video_size is None:
                    video_size = (1080, 1920)
                    logging.info(f"No subtitles, no available images, and no usable default fallback. Video size defaulted to {video_size}.")
                else:
                    logging.info(f"No subtitles, no available images, and no usable default fallback. Using pre-determined video size: {video_size}.")
                logging.warning("Creating a single black ColorClip for the full audio duration as no images are available.")
                clip_to_add = ColorClip(size=video_size, color=(0, 0, 0), duration=segment_duration)
                source_type = "ColorClip (black fallback)"

            if clip_to_add:
                video_segments_for_concatenation.append(clip_to_add)
                logging.debug(f"Prepared single background segment using {source_type} with duration {segment_duration}s.")
            else:
                # This case should ideally not be reached if logic is correct, but as a safeguard:
                if video_size is None: video_size = (1080, 1920) # Ensure video_size is set
                logging.error("Failed to create any background clip in no-subtitle scenario. Adding a default black ColorClip.")
                fallback_clip = ColorClip(size=video_size, color=(0,0,0), duration=segment_duration)
                video_segments_for_concatenation.append(fallback_clip)

        except Exception as e:
            logging.error(f"Error processing background for no-subtitle case (image: '{image_path_for_segment}'): {e}. Using black ColorClip fallback.")
            if 'temp_image_clip' in locals() and clip_to_add != temp_image_clip : temp_image_clip.close() # temp_image_clip might not be defined
            if clip_to_add and clip_to_add not in video_segments_for_concatenation : clip_to_add.close()

            if video_size is None: video_size = (1080, 1920)
            fallback_clip = ColorClip(size=video_size, color=(0, 0, 0), duration=segment_duration)
            video_segments_for_concatenation.append(fallback_clip)
            logging.debug(f"Prepared fallback background segment (ColorClip) with duration {segment_duration}s due to error.")

        return video_segments_for_concatenation, video_size

    # Case 2: Subtitles are present
    num_subs_per_segment = 3
    logging.info(f"Subtitles found. Preparing background segments based on subtitle timings (approx. {num_subs_per_segment} subtitles per segment).")
    for i in range(0, len(subtitles), num_subs_per_segment):
        start_time_sub = subtitles[i].start.ordinal / 1000.0 # Keep for reference, not for clip.with_start()
        end_idx = min(i + num_subs_per_segment, len(subtitles))
        actual_end_time_for_segment_sub = subtitles[end_idx - 1].end.ordinal / 1000.0
        
        duration = actual_end_time_for_segment_sub - start_time_sub
        if duration <= 0:
            logging.warning(f"Calculated non-positive duration {duration:.2f}s for a background segment based on subtitles {subtitles[i].index} to {subtitles[end_idx-1].index}. Skipping this segment.")
            continue

        current_image_index = (current_image_index + 1)
        clip_to_add = None
        current_image_path = None
        source_type = "ColorClip" # Default for logging

        try:
            if available_background_images:
                current_image_path = available_background_images[current_image_index % len(available_background_images)]
                temp_image_clip = ImageClip(current_image_path)
                source_type = f"ImageClip ('{current_image_path}')"
            elif default_fallback_image_path and os.path.isfile(default_fallback_image_path) and os.access(default_fallback_image_path, os.R_OK):
                logging.info(f"No images in {BACKGROUND_IMAGES_DIR} for segment {i//num_subs_per_segment + 1}, using default fallback: {default_fallback_image_path}.")
                current_image_path = default_fallback_image_path
                temp_image_clip = ImageClip(default_fallback_image_path)
                source_type = f"ImageClip (default fallback '{current_image_path}')"
            else: # No images and no default fallback, use ColorClip
                if video_size is None:
                    video_size = (1080, 1920)
                    logging.info(f"Video size not yet determined and no images available for segment {i//num_subs_per_segment + 1}. Defaulting to {video_size}.")
                # else: video_size is already set, use it.
                clip_to_add = ColorClip(size=video_size, color=(0, 0, 0), duration=duration)
                source_type = "ColorClip (black fallback)"
                logging.debug(f"Using {source_type} for background segment {i//num_subs_per_segment + 1} as no images are available.")

            if clip_to_add is None: # This means temp_image_clip was loaded (or attempted)
                if video_size is None:
                    video_size = temp_image_clip.size
                    logging.info(f"Video size determined from '{current_image_path}': {video_size} for segment {i//num_subs_per_segment + 1}.")
                
                if temp_image_clip.size != video_size:
                    logging.debug(f"Resizing image '{current_image_path}' from {temp_image_clip.size} to {video_size} for segment {i//num_subs_per_segment + 1}.")
                    resized_clip = temp_image_clip.resized(video_size)
                    temp_image_clip.close()
                    image_clip_to_use = resized_clip
                else:
                    image_clip_to_use = temp_image_clip
                
                clip_to_add = image_clip_to_use.with_duration(duration)

            video_segments_for_concatenation.append(clip_to_add)
            logging.debug(f"Prepared background segment {i//num_subs_per_segment + 1} using {source_type} with duration {duration:.2f}s.")

        except Exception as e:
            err_msg_path = current_image_path if current_image_path else "the selected image/colorclip"
            logging.error(f"Error with background {err_msg_path} for segment {i//num_subs_per_segment + 1} (duration {duration:.2f}s): {e}. Using black ColorClip fallback.")
            # Ensure any partially created/loaded clips are closed if not added
            if 'temp_image_clip' in locals() and clip_to_add != temp_image_clip: temp_image_clip.close()
            if 'image_clip_to_use' in locals() and clip_to_add != image_clip_to_use: image_clip_to_use.close()
            if clip_to_add and clip_to_add not in video_segments_for_concatenation: clip_to_add.close()

            if video_size is None:
                video_size = (1080, 1920)
                logging.warning(f"Video size was None during error handling for segment {i//num_subs_per_segment + 1}. Defaulted to {video_size}.")
            
            fallback_segment = ColorClip(size=video_size, color=(0, 0, 0), duration=duration)
            video_segments_for_concatenation.append(fallback_segment)
            logging.debug(f"Prepared fallback background segment (ColorClip) for segment {i//num_subs_per_segment + 1} with duration {duration:.2f}s due to error.")
            
    return video_segments_for_concatenation, video_size

def _generate_subtitle_overlay_clips(subtitles, video_size, audio_clip_duration, srt_filepath_logging):
    """
    Generates a list of TextClip objects for subtitles, including a semi-transparent background.
    """
    if video_size is None: # Should be determined by background segments or default
        logging.warning(f"Video size is None for subtitle generation for {srt_filepath_logging}. Defaulting to 1080x1920.")
        video_size = (1080, 1920)

    # Background for subtitles (semi-transparent black bar)
    bg_clip = ColorClip(size=video_size, color=(0, 0, 0), duration=audio_clip_duration)
    bg_clip = bg_clip.with_opacity(0.7)  # Increased opacity from 0.5 to 0.7
    
    subtitle_clips_list = [bg_clip]
    
    if subtitles and len(subtitles) > 0:
        logging.info(f"Generating {len(subtitles)} subtitle clips for {srt_filepath_logging}.")
        for sub in subtitles:
            start_time = sub.start.ordinal / 1000.0
            end_time = sub.end.ordinal / 1000.0
            duration = end_time - start_time
            
            if start_time < audio_clip_duration:
                if sub.text and sub.text.strip():
                    try:
                        subtitle_clip_obj = generate_subtitle_clip(sub.text.strip(), start_time, duration, video_size)
                        subtitle_clips_list.append(subtitle_clip_obj)
                        logging.debug(f"Added subtitle clip for '{srt_filepath_logging}': '{sub.text.strip()[:30]}...'")
                    except Exception as e:
                        logging.error(f"Failed to generate TextClip for subtitle: '{sub.text.strip()[:30]}...' from {srt_filepath_logging}. Error: {e}")
                else:
                    logging.warning(f"Empty subtitle text found at {start_time}-{end_time} in '{srt_filepath_logging}'. Skipping.")
    else:
        logging.info(f"No actual subtitles to process for video overlay for {srt_filepath_logging}.")
        # subtitle_clips_list will only contain the bg_clip, which is fine if there are no subtitles
        
    return subtitle_clips_list

# Step 4: Main video processing function
def create_full_video(audio_path, srt_path, output_path):
    """Create the full-length video with subtitles."""
    logging.info(f"Starting video creation for audio: '{audio_path}', srt: '{srt_path}'")
    if not os.path.isfile(audio_path) or not os.access(audio_path, os.R_OK):
        logging.error(f"Audio file '{audio_path}' does not exist or is not readable.")
        return

    audio_clip = None
    video_segments = [] # Ensure initialized for finally block
    subtitle_clips_list = [] # Ensure initialized for finally block
    video_with_subtitles = None

    try:
        audio_clip = AudioFileClip(audio_path)
        logging.debug(f"Audio file '{audio_path}' loaded successfully by MoviePy.")
        
        if not os.path.isfile(srt_path) or not os.access(srt_path, os.R_OK):
            logging.error(f"SRT file '{srt_path}' does not exist or is not readable.")
            return # audio_clip will be closed in finally

        subtitles_pysrt = pysrt.open(srt_path) # Renamed to avoid conflict
        if not subtitles_pysrt: 
             logging.warning(f"SRT file '{srt_path}' is empty or malformed. Proceeding without subtitles.")
        else:
            logging.debug(f"SRT file '{srt_path}' loaded successfully, containing {len(subtitles_pysrt)} subtitles.")

        available_bg_images = get_background_images() 
        
        if not available_bg_images:
            logging.info(f"No background images found in '{BACKGROUND_IMAGES_DIR}'. Attempting to use default '{DEFAULT_BACKGROUND_IMAGE}'.")
            if os.path.isfile(DEFAULT_BACKGROUND_IMAGE) and os.access(DEFAULT_BACKGROUND_IMAGE, os.R_OK):
                available_bg_images = [DEFAULT_BACKGROUND_IMAGE]
                logging.info(f"Using default background image: '{DEFAULT_BACKGROUND_IMAGE}'")
            else:
                logging.warning(f"Default background '{DEFAULT_BACKGROUND_IMAGE}' not found/readable. Black backgrounds will be used.")
                available_bg_images = []

        prepared_bg_clips, video_size = _prepare_background_segments(subtitles_pysrt, available_bg_images, None, audio_clip.duration, DEFAULT_BACKGROUND_IMAGE if not available_bg_images else None)

        if video_size is None: # If _prepare_background_segments couldn't determine it
            video_size = (1080, 1920) # Default
            logging.warning(f"Video size could not be determined by _prepare_background_segments; defaulting to {video_size} for '{audio_path}'.")

        # Handle empty or invalid prepared_bg_clips
        if not prepared_bg_clips:
            logging.warning(f"No background clips were prepared by _prepare_background_segments for '{audio_path}'. Creating a single black ColorClip for the full audio duration.")
            if video_size is None: # Should have been set above, but as a fallback
                video_size = (1080, 1920)
                logging.warning(f"Video size was still None when creating fallback background. Defaulted to {video_size}.")
            fallback_bg_clip = ColorClip(size=video_size, color=(0,0,0), duration=audio_clip.duration)
            prepared_bg_clips = [fallback_bg_clip]
            
        # Calculate current total duration of background clips and adjust the last clip
        if prepared_bg_clips: # Should always be true now due to the above fallback
            total_bg_duration = sum(clip.duration for clip in prepared_bg_clips if clip.duration is not None)
            logging.debug(f"Initial total_bg_duration: {total_bg_duration:.2f}s for {len(prepared_bg_clips)} segments.")

            duration_difference = audio_clip.duration - total_bg_duration
            
            if abs(duration_difference) > 0.01: # Only adjust if difference is meaningful
                last_clip = prepared_bg_clips[-1]
                original_last_clip_duration = last_clip.duration if last_clip.duration is not None else 0
                new_last_clip_duration = original_last_clip_duration + duration_difference
                
                logging.info(f"Adjusting last background clip duration. Original total: {total_bg_duration:.2f}s, Target audio: {audio_clip.duration:.2f}s, Difference: {duration_difference:.2f}s.")
                logging.info(f"Last clip original duration: {original_last_clip_duration:.2f}s, New proposed duration: {new_last_clip_duration:.2f}s.")

                if new_last_clip_duration <= 0:
                    logging.warning(f"New last clip duration ({new_last_clip_duration:.2f}s) is zero or negative. Setting to a small positive value (0.04s).")
                    new_last_clip_duration = 0.04 # 1 frame at 25fps
                
                # Create a new clip with the adjusted duration
                try:
                    # For ColorClip, we need to create a new instance with the new duration
                    if isinstance(last_clip, ColorClip):
                        adjusted_last_clip = ColorClip(size=last_clip.size, color=last_clip.color, duration=new_last_clip_duration)
                    else:
                        adjusted_last_clip = last_clip.with_duration(new_last_clip_duration)
                    
                    # Close the old last_clip if it's not the same object and has a close method
                    if last_clip != adjusted_last_clip and hasattr(last_clip, 'close') and callable(last_clip.close):
                        last_clip.close() 
                        
                    prepared_bg_clips[-1] = adjusted_last_clip
                    logging.info(f"Last background clip duration adjusted to: {new_last_clip_duration:.2f}s.")
                    # Recalculate for logging
                    final_total_bg_duration = sum(clip.duration for clip in prepared_bg_clips if clip.duration is not None)
                    logging.debug(f"Final total_bg_duration after adjustment: {final_total_bg_duration:.2f}s.")
                except Exception as e:
                    logging.error(f"Error adjusting duration for the last clip: {e}. Using original last clip.")
            else:
                logging.debug(f"No significant duration adjustment needed for background clips (difference: {duration_difference:.2f}s).")
        else: # This case should ideally not be reached if the fallback logic is correct
            logging.error("prepared_bg_clips is unexpectedly empty even after fallback. Cannot create background track.")
            return

        # Concatenate background clips into a single track
        final_background_track = None
        if prepared_bg_clips:
            try:
                # Filter out None durations just in case, though new_last_clip_duration should be positive.
                valid_clips_for_concat = [c for c in prepared_bg_clips if c.duration is not None and c.duration > 0]
                if valid_clips_for_concat:
                    final_background_track = concatenate_videoclips(valid_clips_for_concat, method="chain")
                    logging.debug(f"Background clips concatenated successfully into a track of duration: {final_background_track.duration:.2f}s.")
                else:
                    logging.error("No valid background clips with positive duration to concatenate. Cannot create background track.")
                    # Create a full duration black clip as ultimate fallback if concatenation fails
                    if video_size is None: video_size = (1080, 1920)
                    final_background_track = ColorClip(size=video_size, color=(0,0,0), duration=audio_clip.duration)
                    logging.warning("Using a full duration black ColorClip as final_background_track due to concatenation issues.")

            except Exception as e:
                logging.error(f"Error during background clip concatenation: {e}. Using a fallback black background.")
                if video_size is None: video_size = (1080, 1920)
                final_background_track = ColorClip(size=video_size, color=(0,0,0), duration=audio_clip.duration)
        
        if not final_background_track: # Should be set by now
            logging.error("final_background_track is None. Aborting video creation.")
            # Close individual prepared_bg_clips if final_background_track wasn't made
            for clip in prepared_bg_clips:
                if hasattr(clip, 'close') and callable(clip.close):
                    clip.close()
            return

        subtitle_clips_list = _generate_subtitle_overlay_clips(subtitles_pysrt, video_size, audio_clip.duration, srt_path)
        
        logging.debug(f"Compositing final background track and {len(subtitle_clips_list)} subtitle overlay clips for '{audio_path}'.")
        # The final_background_track is already set to the full audio_clip.duration (or should be)
        # Subtitle clips have their own start times and durations.
        final_clips_for_composition = [final_background_track] + subtitle_clips_list
        
        video_with_subtitles = CompositeVideoClip(final_clips_for_composition, size=video_size).with_duration(audio_clip.duration)
        video_with_subtitles = video_with_subtitles.with_audio(audio_clip)
        
        logging.info(f"Starting video file write for: '{output_path}'")
        video_with_subtitles.write_videofile(
            output_path,
            codec="libx264",
            fps=24,
            audio_codec="aac",
            threads=4,
            preset='faster',
            bitrate="5000k",
            logger='bar' 
        )
        logging.info(f"Successfully wrote video file to '{output_path}'")

    except (OSError, RuntimeError) as e: # Catch MoviePy specific operational errors
        logging.error(f"MoviePy related error during video creation for '{audio_path}': {str(e)}")
    except Exception as e: # Catch any other unexpected error
        logging.exception(f"Unexpected error during full video creation for '{audio_path}':")
    finally:
        # Clean up
        if 'video_with_subtitles' in locals() and video_with_subtitles: 
            video_with_subtitles.close()
        if 'audio_clip' in locals() and audio_clip: 
            audio_clip.close()
        
        # Clean up individual clips that weren't part of the final composition
        if 'final_background_track' in locals() and final_background_track:
            if not video_with_subtitles or (video_with_subtitles and final_background_track not in getattr(video_with_subtitles, 'clips', [])):
                if hasattr(final_background_track, 'close') and callable(final_background_track.close):
                    logging.debug("Closing final_background_track as it's not in the final composite video.")
                    final_background_track.close()
        
        # Clean up subtitle clips
        for clip_obj in subtitle_clips_list:
            if clip_obj and (not video_with_subtitles or clip_obj not in getattr(video_with_subtitles, 'clips', [])):
                if hasattr(clip_obj, 'close') and callable(clip_obj.close):
                    clip_obj.close()

def sanitize_filename(title):
    """Convert a title to a safe filename."""
    # Remove or replace characters that are not safe for filenames
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
    # Limit length to avoid too long filenames
    safe_title = safe_title[:100]
    return safe_title.strip()

# New function for clip adjustment prompt
def prompt_for_clip_adjustment(clip_info, video_duration_seconds, srt_path):
    """
    Prompts the user to adjust, keep, or skip a clip.
    Includes context from the SRT file.
    """
    title = clip_info.get('title', 'N/A')
    original_start_time_str = clip_info.get('start_time', 'N/A')
    original_end_time_str = clip_info.get('end_time', 'N/A')
    clip_content = clip_info.get('content', 'N/A')

    text_before_clip = "No text found immediately before."
    text_after_clip = "No text found immediately after."

    if srt_path and os.path.exists(srt_path):
        try:
            subs = pysrt.open(srt_path, encoding='utf-8')
            clip_start_seconds = parse_timestamp(original_start_time_str)
            clip_end_seconds = parse_timestamp(original_end_time_str)

            if clip_start_seconds is not None:
                for sub in reversed(subs):
                    sub_end_seconds = sub.end.ordinal / 1000.0
                    if sub_end_seconds <= clip_start_seconds:
                        text_before_clip = sub.text
                        break
            
            if clip_end_seconds is not None:
                for sub in subs:
                    sub_start_seconds = sub.start.ordinal / 1000.0
                    if sub_start_seconds >= clip_end_seconds:
                        text_after_clip = sub.text
                        break
        except Exception as e:
            logging.warning(f"Error processing SRT file '{srt_path}' for context: {e}")

    print(f"\n--- Adjust Clip ---")
    print(f"Text Before Clip: {text_before_clip}")
    print(f"Title: {title}")
    print(f"Current Start: {original_start_time_str}")
    print(f"Current End:   {original_end_time_str}")
    print(f"Clip Content: {clip_content}")
    print(f"Text After Clip: {text_after_clip}")

    if video_duration_seconds > 0:
        print(f"Video Duration: {format_timedelta(video_duration_seconds)}")
    else:
        print(f"Video Duration: Not available")

    while True:
        action = input("Choose action: [k]eep, [a]djust, [s]kip clip? (k/a/s): ").lower().strip()
        if action in ['k', 'a', 's']:
            break
        print("Invalid input. Please enter 'k', 'a', or 's'.")

    if action == 'k':
        logging.info(f"User chose to KEEP clip: '{title}'.")
        print(f"Keeping clip '{title}' as is.")
        return clip_info
    elif action == 's':
        logging.info(f"User chose to SKIP clip: '{title}'.")
        print(f"Skipping clip '{title}'.")
        return None
    elif action == 'a':
        # Logging for 'adjust' will be done after successful validation
        print(f"Adjusting clip '{title}':")
        while True:
            new_start_input_str = input(f"Enter new start time (current: {original_start_time_str}, format HH:MM:SS,mmm): ").strip()
            new_end_input_str = input(f"Enter new end time (current: {original_end_time_str}, format HH:MM:SS,mmm): ").strip()

            new_start_seconds = parse_timestamp(new_start_input_str)
            new_end_seconds = parse_timestamp(new_end_input_str)

            if new_start_seconds is None or new_end_seconds is None:
                logging.warning(f"Invalid time format entered by user for clip '{title}': Start='{new_start_input_str}', End='{new_end_input_str}'.")
                print("Invalid time format. Please use HH:MM:SS,mmm (e.g., 00:01:23,456).")
                # parse_timestamp already logs an error at ERROR level if parsing fails,
                # so the WARNING here is for the user input attempt.
                continue

            if new_start_seconds < 0:
                logging.warning(f"Time validation failed for clip '{title}': Start time {new_start_seconds}s is negative. Input: '{new_start_input_str}'.")
                print("Start time cannot be negative.")
                continue
            
            if video_duration_seconds > 0 and new_end_seconds > video_duration_seconds: # only check if video_duration is valid
                logging.warning(f"Time validation failed for clip '{title}': End time {new_end_seconds}s (Input: '{new_end_input_str}') exceeds video duration {video_duration_seconds}s.")
                print(f"End time ({format_timedelta(new_end_seconds)}) cannot exceed video duration ({format_timedelta(video_duration_seconds)}).")
                continue
            
            if new_start_seconds >= new_end_seconds:
                logging.warning(f"Time validation failed for clip '{title}': Start time {new_start_seconds}s (Input: '{new_start_input_str}') is not before end time {new_end_seconds}s (Input: '{new_end_input_str}').")
                print(f"Start time ({format_timedelta(new_start_seconds)}) must be before end time ({format_timedelta(new_end_seconds)}).")
                continue

            # All validations passed
            logging.info(f"User chose to ADJUST clip: '{title}'. Original Start: {original_start_time_str}, Original End: {original_end_time_str}.")
            clip_info['start_time'] = format_timedelta(new_start_seconds)
            clip_info['end_time'] = format_timedelta(new_end_seconds)
            
            # Remove temporary keys if they were added from a previous version of the code or different logic path
            clip_info.pop('new_start_time', None)
            clip_info.pop('new_end_time', None)

            logging.info(f"Clip '{title}' adjusted. Original: {original_start_time_str}->{original_end_time_str}, New: {clip_info['start_time']}->{clip_info['end_time']}")
            print(f"Clip '{title}' adjusted. New Start: {clip_info['start_time']}, New End: {clip_info['end_time']}")
            return clip_info

# Step 5: Clip extraction function
def extract_clips(video_path, clips_data, output_dir='clips'):
    """Extract clips from video using timestamps."""
    """Extract clips from video using timestamps."""
    logging.info(f"Starting clip extraction for video: '{video_path}' into directory: '{output_dir}'")
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.debug(f"Output directory '{output_dir}' ensured.")
    except OSError as e:
        logging.error(f"Error creating output directory '{output_dir}': {str(e)}. Please check permissions and disk space.")
        return 

    # Validate video_path
    if not os.path.isfile(video_path) or not os.access(video_path, os.R_OK):
        logging.error(f"Video file for clipping '{video_path}' does not exist or is not readable.")
        return

    video = None
    try:
        video = VideoFileClip(video_path)
        logging.debug(f"Video file '{video_path}' loaded successfully for clipping.")
    except (OSError, RuntimeError) as e:
        logging.error(f"MoviePy error loading video file '{video_path}' for clipping: {str(e)}. Ensure FFmpeg is installed and the video file is valid.")
        return
    except Exception as e:
        logging.exception(f"Unexpected error loading video file '{video_path}' for clipping with MoviePy:")
        return
    
    try:
        if not isinstance(clips_data, dict) or 'clips' not in clips_data or not isinstance(clips_data['clips'], list):
            logging.error(f"Invalid clips_data structure passed to extract_clips for video '{video_path}'. Expected dict with a list of clips.")
            if video: video.close()
            return

        logging.info(f"Processing {len(clips_data['clips'])} clip(s) for '{video_path}'.")
        for i, clip_data_item in enumerate(clips_data['clips'], 1): 
            try:
                # Basic validation, detailed validation is expected to be done in main() before calling this.
                if not all(k in clip_data_item for k in ('start_time', 'end_time', 'title')):
                    logging.warning(f"Clip {i} in '{video_path}' is missing critical keys (start_time, end_time, title). Skipping.")
                    continue

                start_time_val = parse_timestamp(clip_data_item['start_time'])
                end_time_val = parse_timestamp(clip_data_item['end_time'])

                if start_time_val is None or end_time_val is None:
                    logging.warning(f"Invalid timestamps for clip {i} ('{clip_data_item['title']}') in '{video_path}'. Start: {clip_data_item['start_time']}, End: {clip_data_item['end_time']}. Skipping.")
                    continue
                
                if end_time_val <= start_time_val:
                    logging.warning(f"End time not after start time for clip {i} ('{clip_data_item['title']}') in '{video_path}'. Start: {clip_data_item['start_time']}, End: {clip_data_item['end_time']}. Skipping.")
                    continue
                
                logging.info(f"Processing clip {i}: '{clip_data_item['title']}' from {clip_data_item['start_time']} to {clip_data_item['end_time']} for '{video_path}'.")
                
                sub_clip = None
                try:
                    sub_clip = video.subclipped(start_time_val, end_time_val)
                    safe_title = sanitize_filename(clip_data_item['title'])
                    output_clip_path = os.path.join(output_dir, f'{safe_title}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
                    
                    logging.info(f"Writing clip '{safe_title}' to '{output_clip_path}'...")
                    sub_clip.write_videofile(
                        output_clip_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a', 
                        remove_temp=True,
                        logger='bar' # Using 'bar' for progress, can be None for less verbosity
                    )
                    logging.info(f"Successfully saved clip {i} ('{safe_title}') to '{output_clip_path}'")
                except (OSError, RuntimeError) as e:
                    logging.error(f"MoviePy error processing or writing subclip {i} ('{clip_data_item.get('title', 'Untitled')}'): {str(e)}. This could be due to FFmpeg issues, invalid timestamps, or disk space problems.")
                except Exception as e:
                    logging.exception(f"Unexpected error while processing subclip {i} ('{clip_data_item.get('title', 'Untitled')}'):")
                finally:
                    if sub_clip: sub_clip.close() 
            
            except Exception as e: 
                logging.exception(f"Unexpected error processing clip data for item {i} ('{clip_data_item.get('title', 'Untitled')}') in '{video_path}':")
    
    finally:
        try:
            if video: video.close() 
        except Exception as e: 
            logging.exception(f"Error during cleanup (closing main video file for '{video_path}'):")

# Step 6: Utility function for audio file management
def get_audio_files(directory=AUDIO_FILES_DIR):
    """Get all audio files from the specified directory."""
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg']
    audio_files = []
    logging.debug(f"Searching for audio files in directory: '{directory}'")

    if not os.path.exists(directory):
        logging.info(f"Audio directory '{directory}' does not exist. Creating it...")
        try:
            os.makedirs(directory)
            logging.info(f"Successfully created audio directory: '{directory}'")
            return []
        except OSError as e:
            logging.error(f"Could not create audio directory '{directory}': {str(e)}. Please check permissions or create the directory manually.")
            return []

    if not os.access(directory, os.R_OK | os.X_OK): 
        logging.error(f"Audio directory '{directory}' is not readable or accessible.")
        return []

    try:
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if any(file_name.lower().endswith(ext) for ext in audio_extensions):
                if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                    audio_files.append(file_path)
                    logging.debug(f"Found readable audio file: '{file_path}'")
                elif not os.path.isfile(file_path):
                    logging.warning(f"Found potential audio item '{file_path}' but it is not a file. Skipping.")
                elif not os.access(file_path, os.R_OK):
                     logging.warning(f"Found audio file '{file_path}' but it is not readable. Skipping.")
    except OSError as e:
        logging.error(f"Error accessing contents of audio directory '{directory}': {str(e)}")
        return []
    
    return audio_files

# Step 7: Main execution flow
def setup_logging():
    """Configures the logging for the script."""
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s")
    
    # File handler
    file_handler = logging.FileHandler("processing.log", mode='a') # Append mode
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO) # Or logging.DEBUG for more verbose console output if needed
    
    # Root logger configuration
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

def main():
    setup_logging()
    logging.info("Script execution started.")
    audio_files = get_audio_files() 
    
    if not audio_files:
        logging.warning(f"No audio files found in the '{AUDIO_FILES_DIR}' directory. Please add audio files and try again.")
        return
    
    logging.info(f"Found {len(audio_files)} audio file(s) to process: {', '.join(map(os.path.basename, audio_files))}")
    
    successful_files = 0
    failed_files = 0

    for audio_path in audio_files:
        if _process_single_audio_file(audio_path):
            successful_files += 1
        else:
            failed_files +=1
            
    logging.info(f"--- All audio file processing complete ---")
    logging.info(f"Successfully processed: {successful_files} file(s).")
    logging.info(f"Failed to process: {failed_files} file(s).")


def _process_single_audio_file(audio_path):
    """
    Processes a single audio file: transcription, full video, transcript analysis, and clip extraction.
    Returns True if processing was successful (or mostly successful), False otherwise.
    """
    audio_file_name = os.path.basename(audio_path)
    logging.info(f"--- Starting processing for audio file: {audio_file_name} ---")
    try:
        base_name = os.path.splitext(audio_file_name)[0]
        output_dir = os.path.join(OUTPUT_DIR_BASE, base_name)
        srt_path = os.path.join(output_dir, f"{base_name}_transcript.srt")
        skip_transcription = False

        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"Output directory '{output_dir}' ensured for {audio_file_name}.")
        except OSError as e:
            logging.error(f"Error creating output directory '{output_dir}' for {audio_file_name}: {str(e)}. Skipping this file.")
            return False

        if os.path.exists(srt_path) and os.access(srt_path, os.R_OK):
            logging.info(f"SRT file '{srt_path}' already exists. Skipping transcription and SRT creation.")
            skip_transcription = True
        else:
            # Step 1: Transcription
            logging.info(f"Step 1: Creating transcript for {audio_file_name}...")
            try:
                model = whisper.load_model(DEFAULT_WHISPER_MODEL)
                transcription_result = model.transcribe(audio_path)
                logging.info(f"Transcription completed for {audio_file_name} using model '{DEFAULT_WHISPER_MODEL}'.")
            except Exception as e:
                logging.exception(f"Error during transcription for {audio_file_name}. Skipping this file.")
                return False
            
            # Step 2: Create SRT file (output_dir is already created)
            try:
                create_srt(transcription_result["segments"], srt_path)
                logging.info(f"SRT transcript saved to '{srt_path}' for {audio_file_name}.")
            except Exception as e:
                logging.exception(f"Error creating SRT file '{srt_path}' for {audio_file_name}. Skipping this file.")
                return False

        # Step 3: Create full-length video
        logging.info(f"Step 2 (was 3): Creating full-length video for {audio_file_name}...") # Corrected step numbering in log
        output_video_path = os.path.join(output_dir, f"{base_name}_video.mp4")

        if os.path.exists(output_video_path) and os.access(output_video_path, os.R_OK):
            logging.info(f"Full video file '{output_video_path}' already exists. Skipping video creation.")
        else:
            create_full_video(audio_path, srt_path, output_video_path) 
        
        if not os.path.exists(output_video_path):
            logging.error(f"Full video creation failed for '{audio_path}'. Skipping clip extraction.")
            return False # Video creation is critical for clip extraction
        logging.info(f"Full-length video created at '{output_video_path}' for {audio_file_name}.")

        # Step 4: Analyze transcript / Load existing clips JSON
        logging.info(f"Step 3 (was 4): Analyzing transcript or loading existing clips JSON for {audio_file_name}...")
        clips_json_path = os.path.join(output_dir, f"{base_name}_clips.json")
        clips_to_process = None
        skip_analysis_and_json_creation = False

        if os.path.exists(clips_json_path) and os.access(clips_json_path, os.R_OK):
            logging.info(f"Clips JSON file '{clips_json_path}' already exists. Loading clips from file.")
            try:
                with open(clips_json_path, 'r', encoding='utf-8') as f:
                    clips_to_process = json.load(f)
                logging.info(f"Successfully loaded clips from '{clips_json_path}'.")
                # Basic validation of loaded data
                if not isinstance(clips_to_process, dict) or 'clips' not in clips_to_process or not isinstance(clips_to_process.get('clips'), list):
                    logging.warning(f"Loaded clips JSON from '{clips_json_path}' has invalid structure. Will attempt to regenerate.")
                    clips_to_process = None # Invalidate to trigger regeneration
                else:
                    skip_analysis_and_json_creation = True
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from existing file '{clips_json_path}': {str(e)}. Will attempt to regenerate.")
                clips_to_process = None # Invalidate to trigger regeneration
            except Exception as e:
                logging.exception(f"Unexpected error loading clips JSON from '{clips_json_path}'. Will attempt to regenerate.")
                clips_to_process = None # Invalidate to trigger regeneration
        
        if not skip_analysis_and_json_creation:
            logging.info(f"Proceeding with transcript analysis for {audio_file_name}.")
            transcript_content = ""
            try:
                with open(srt_path, 'r', encoding='utf-8') as file:
                    transcript_content = file.read()
                logging.debug(f"Successfully read SRT file '{srt_path}' for analysis.")
            except Exception as e:
                logging.exception(f"Error reading SRT file '{srt_path}' for analysis. Skipping clip selection and extraction for this file.")
                return False # Transcript content is critical for clip selection

            analysis_result = analyze_transcript(transcript_content)
            if analysis_result is None:
                logging.error(f"Failed to analyze transcript from '{srt_path}' for {audio_file_name}. Skipping clip extraction.")
                return True # Partial success: video created, but no clips.

            clips_data = None
            try:
                cleaned_result = clean_json_response(analysis_result)
                clips_data = json.loads(cleaned_result)
                logging.debug(f"Successfully parsed JSON response from Gemini API for {audio_file_name}.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from Gemini for {audio_file_name}: {str(e)}. Raw: {cleaned_result}")
                return True 
            except Exception as e:
                logging.exception(f"Unexpected error parsing Gemini API response for {audio_file_name}:")
                return True

            if not isinstance(clips_data, dict) or 'clips' not in clips_data or not isinstance(clips_data.get('clips'), list):
                logging.error(f"Invalid JSON structure from Gemini for {audio_file_name}. Data: {clips_data}")
                return True
            if not clips_data['clips']:
                logging.info(f"No clips identified by Gemini for {audio_file_name}.")
                return True
                
            validated_clips = []
            logging.debug(f"Validating {len(clips_data['clips'])} clips from Gemini for {audio_file_name}.")
            for i, clip_info in enumerate(clips_data['clips'], 1):
                if not (isinstance(clip_info, dict) and all(k in clip_info for k in ['start_time', 'end_time', 'title', 'content'])):
                    logging.warning(f"Clip item {i} for {audio_file_name} is malformed. Skipping. Data: {clip_info}")
                    continue
                start_seconds = parse_timestamp(clip_info['start_time'])
                end_seconds = parse_timestamp(clip_info['end_time'])
                if start_seconds is None or end_seconds is None or end_seconds <= start_seconds:
                    logging.warning(f"Clip item {i} for {audio_file_name} has invalid/illogical timestamps. Skipping. Start: {clip_info['start_time']}, End: {clip_info['end_time']}")
                    continue
                validated_clips.append(clip_info)

            if not validated_clips:
                logging.info(f"No valid clips found for {audio_file_name} after validation.")
                return True 
            
            clips_to_process = {'clips': validated_clips}
            logging.info(f"Validated {len(validated_clips)} clips for {audio_file_name}.")
            
            # JSON saving will be done after adjustment phase

        # Step 5: Clip Adjustment and then Saving JSON / Creating short clips
        # Ensure clips_to_process is valid before proceeding with adjustment
        if clips_to_process is None or not clips_to_process.get('clips'):
            logging.warning(f"No valid clips (either loaded or generated) available for {audio_file_name}. Skipping clip extraction.")
            return True # Partial success as main video might be done.

        # --- Add clip adjustment phase ---
        logging.info(f"Starting clip adjustment phase for {audio_file_name}...")
        video_duration = 0
        try:
            full_video_clip_for_duration = VideoFileClip(output_video_path)
            video_duration = full_video_clip_for_duration.duration
            full_video_clip_for_duration.close()
            logging.info(f"Full video duration for {audio_file_name} is {video_duration:.2f} seconds.")
        except Exception as e:
            logging.error(f"Error getting video duration for {output_video_path}: {e}. Cannot proceed with interactive clip adjustment.")
            # Decide if this is fatal for clip extraction or if we proceed with unadjusted clips
            # For now, let's proceed with unadjusted clips if duration cannot be obtained.
            # If video_duration remains 0, prompt_for_clip_adjustment will show that.

        adjusted_clips = []
        num_original_clips = 0
        if clips_to_process and 'clips' in clips_to_process: # Check if clips_to_process is not None
            num_original_clips = len(clips_to_process['clips'])
            for clip_item in clips_to_process['clips']:
                modified_clip = prompt_for_clip_adjustment(clip_item, video_duration, srt_path) # Pass srt_path
                if modified_clip:
                    adjusted_clips.append(modified_clip)
            clips_to_process['clips'] = adjusted_clips
            num_final_clips = len(adjusted_clips)
            logging.info(f"Clip adjustment phase completed for {audio_file_name}. Original clips: {num_original_clips}, Final clips: {num_final_clips}.")
            
            # Save the possibly modified clips_to_process to JSON
            # This will also save if adjusted_clips is empty (user skipped all)
            if clips_to_process: # Ensure clips_to_process itself is not None
                try:
                    with open(clips_json_path, 'w') as f:
                        json.dump(clips_to_process, f, indent=2)
                    logging.info(f"Saved final (possibly adjusted) clips JSON to '{clips_json_path}'.")
                except Exception as e:
                    logging.exception(f"Error saving final clips JSON for {audio_file_name}:")
                    # Decide if this is critical. For now, non-critical, proceed if possible.
            else:
                logging.warning(f"Skipping JSON save for {audio_file_name} as clips_to_process is None (should not happen if loaded/generated correctly).")

        else: # This 'else' corresponds to 'if clips_to_process and 'clips' in clips_to_process:'
            logging.warning(f"No clips to adjust for {audio_file_name} (clips_to_process or 'clips' key was missing before adjustment).")


        if not clips_to_process or not clips_to_process.get('clips'): # Re-check after adjustment and potential save
            logging.warning(f"No clips remaining after adjustment for {audio_file_name}. Skipping clip extraction.")
            return True


        logging.info(f"Step 4 (was 5): Creating short clips for {audio_file_name}...")
        clips_dir = os.path.join(output_dir, 'clips')
        extract_clips(output_video_path, clips_to_process, clips_dir)
        logging.info(f"Finished creating short clips for {audio_file_name}.")
        return True # All main steps completed for this file

    except Exception as e:
        logging.exception(f"--- MAJOR UNEXPECTED ERROR processing {audio_file_name} ---")
        return False
    finally:
        logging.info(f"--- Finished processing for audio file: {audio_file_name} ---")

if __name__ == "__main__":
    main()
    logging.info("Script execution finished.")
