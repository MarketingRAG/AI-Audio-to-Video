# Step 1: Import the necessary libraries
import whisper
import os
from datetime import timedelta
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip, ColorClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import pysrt
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re
from datetime import datetime

# Step 2: Configure the LLM API
# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

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
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content: {str(e)}")
        return None

def parse_timestamp(timestamp):
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    hours, minutes, seconds = timestamp.split(':')
    seconds, milliseconds = seconds.split(',')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def generate_subtitle_clip(text, start_time, duration, video_size):
    """Generate a subtitle clip with the given text and timing."""
    txt_clip = TextClip(
        text=text,
        size=(int(video_size[0] * 0.85), int(video_size[1] * 0.2)),
        color='#C8102E',
        font="/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        stroke_color='#FAFAFA',
        stroke_width=4,
        method='caption'
    )
    txt_clip = txt_clip.with_duration(duration).with_start(start_time)
    txt_clip = txt_clip.with_position(('center', 'center'))
    return txt_clip

def get_background_images(directory='background_images'):
    """Get all image files from the specified directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_files = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Creating it...")
        os.makedirs(directory)
        return []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory, file))
    
    return image_files

# Step 4: Main video processing function
def create_full_video(audio_path, srt_path, output_path):
    """Create the full-length video with subtitles."""
    # Load the audio file
    audio_clip = AudioFileClip(audio_path)
    
    # Get background images
    background_images = get_background_images()
    if not background_images:
        print("No background images found. Using default background.jpg")
        background_images = ["background.jpg"]
    
    # Load the subtitles
    subtitles = pysrt.open(srt_path)
    video_size = None
    
    # Create video segments with different background images
    video_segments = []
    current_image_index = -1  # Start at -1 so first increment gives us 0
    
    # First, create all the background segments that span multiple subtitle segments
    for i in range(0, len(subtitles), 3):
        current_image_index = (current_image_index + 1) % len(background_images)
        
        # Calculate the total duration of this background (3 segments or remaining segments)
        start_time = subtitles[i].start.ordinal / 1000.0
        end_idx = min(i + 3, len(subtitles))
        end_time = subtitles[end_idx - 1].end.ordinal / 1000.0
        duration = end_time - start_time
        
        # Load and prepare the background image
        image_clip = ImageClip(background_images[current_image_index])
        if video_size is None:
            video_size = image_clip.size
        if image_clip.size != video_size:
            image_clip = image_clip.resized(video_size)
        
        # Create the background segment spanning multiple subtitle segments
        segment = image_clip.with_duration(duration).with_start(start_time)
        video_segments.append(segment)
    
    # Create background for subtitles
    bg_clip = ColorClip(size=video_size, color=(0, 0, 0))
    bg_clip = bg_clip.with_opacity(0.5)
    bg_clip = bg_clip.with_duration(audio_clip.duration)
    
    subtitle_clips = [bg_clip]
    
    # Generate subtitle clips
    for sub in subtitles:
        start_time = sub.start.ordinal / 1000.0
        end_time = sub.end.ordinal / 1000.0
        duration = end_time - start_time
        
        if start_time < audio_clip.duration:
            subtitle_clip = generate_subtitle_clip(sub.text, start_time, duration, video_size)
            subtitle_clips.append(subtitle_clip)
    
    # Create final video by compositing all clips
    video_with_subtitles = CompositeVideoClip(
        video_segments + subtitle_clips
    ).with_duration(audio_clip.duration)
    
    # Add audio
    video_with_subtitles = video_with_subtitles.with_audio(audio_clip)
    
    # Write the video file
    video_with_subtitles.write_videofile(
        output_path,
        codec="libx264",
        fps=24,
        audio_codec="aac",
        threads=4,
        preset='faster',
        bitrate="5000k"
    )
    
    # Clean up
    video_with_subtitles.close()
    audio_clip.close()
    for segment in video_segments:
        segment.close()
    for clip in subtitle_clips:
        clip.close()

def sanitize_filename(title):
    """Convert a title to a safe filename."""
    # Remove or replace characters that are not safe for filenames
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
    # Limit length to avoid too long filenames
    safe_title = safe_title[:100]
    return safe_title.strip()

# Step 5: Clip extraction function
def extract_clips(video_path, clips_data, output_dir='clips'):
    """Extract clips from video using timestamps."""
    os.makedirs(output_dir, exist_ok=True)
    video = VideoFileClip(video_path)
    
    try:
        for i, clip in enumerate(clips_data['clips'], 1):
            try:
                start_time = parse_timestamp(clip['start_time'])
                end_time = parse_timestamp(clip['end_time'])
                
                print(f"\nProcessing clip {i} ({clip['start_time']} to {clip['end_time']})...")
                
                # Create subclip using moviepy
                clip_video = video.subclipped(start_time, end_time)
                safe_title = sanitize_filename(clip['title'])
                output_path = os.path.join(output_dir, f'{safe_title}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
                
                print(f"Writing clip {i} to {output_path}...")
                clip_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    logger=None
                )
                
                clip_video.close()
                print(f"Successfully saved clip {i} to {output_path}")
                
            except Exception as e:
                print(f"Error processing clip {i}: {str(e)}")
                continue
    
    finally:
        try:
            video.close()
        except:
            pass

# Step 6: Utility function for audio file management
def get_audio_files(directory='audio_file'):
    """Get all audio files from the specified directory."""
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg']
    audio_files = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Creating it...")
        os.makedirs(directory)
        return []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(os.path.join(directory, file))
    
    return audio_files

# Step 7: Main execution flow
def main():
    # Get audio files from the audio_file directory
    audio_files = get_audio_files()
    
    if not audio_files:
        print("No audio files found in the audio_file directory. Please add audio files and try again.")
        return
    
    # Process each audio file
    for audio_path in audio_files:
        print(f"\nProcessing audio file: {os.path.basename(audio_path)}")
        
        # Create transcript
        print("Step 1: Creating transcript...")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        
        # Create output directory for this audio file
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join('output', base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        srt_path = os.path.join(output_dir, f"{base_name}_transcript.srt")
        create_srt(result["segments"], srt_path)
        
        # Create full-length video
        print("\nStep 2: Creating full-length video...")
        output_video_path = os.path.join(output_dir, f"{base_name}_video.mp4")
        create_full_video(audio_path, srt_path, output_video_path)
        
        # Select clips
        print("\nStep 3: Analyzing transcript and selecting clips...")
        with open(srt_path, 'r', encoding='utf-8') as file:
            transcript = file.read()
        
        result = analyze_transcript(transcript)
        if result is None:
            print("Failed to analyze transcript")
            continue
        
        cleaned_result = clean_json_response(result)
        clips = json.loads(cleaned_result)
        
        print("\nViral-worthy clips identified:")
        for i, clip in enumerate(clips['clips'], 1):
            print(f"\nClip {i}:")
            print(f"Time: {clip['start_time']} - {clip['end_time']}")
            print(f"Content: {clip['content']}")
            print(f"Title: {clip['title']}")
        
        clips_json_path = os.path.join(output_dir, f"{base_name}_clips.json")
        with open(clips_json_path, 'w') as f:
            json.dump(clips, f, indent=2)
        
        # Create shorts
        print("\nStep 4: Creating short clips...")
        clips_dir = os.path.join(output_dir, 'clips')
        extract_clips(output_video_path, clips, clips_dir)

if __name__ == "__main__":
    main()
