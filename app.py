import streamlit as st
import os
import tempfile
from datetime import datetime
# Assuming audio_to_youtube_shorts is now structured with a process_media_file function
from audio_to_youtube_shorts import process_media_file, setup_logging

# Setup logging for the main app
setup_logging()

st.title("Video/Audio Clip Generator")

st.write("Upload an audio or video file to generate potential short clips.")

# Define a directory to store all session uploads and their outputs
# This helps in organizing outputs if multiple files are processed in one session
SESSION_BASE_DIR = os.path.join(tempfile.gettempdir(), "clip_generator_sessions")
if not os.path.exists(SESSION_BASE_DIR):
    os.makedirs(SESSION_BASE_DIR, exist_ok=True)

# Create a unique directory for this specific run/session to store its outputs
# This helps avoid conflicts if the app is run multiple times or by multiple users (on a shared system)
# For a single user session, this might be simpler, but good practice for broader use.
# For this iteration, let's use a timestamped session directory.
# A more robust solution might involve actual session state management if needed.
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
CURRENT_RUN_OUTPUT_BASE_DIR = os.path.join(SESSION_BASE_DIR, RUN_ID)
os.makedirs(CURRENT_RUN_OUTPUT_BASE_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Choose a file (audio or video)", type=["mp3", "wav", "m4a", "ogg", "mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write("File Details:", file_details)

    # Save the uploaded file to a temporary path within the current run's output directory
    # This makes it accessible to the processing script
    temp_input_path = os.path.join(CURRENT_RUN_OUTPUT_BASE_DIR, uploaded_file.name)

    try:
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully to a temporary path.")
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        # Stop further processing if file saving fails
        st.stop()


    if st.button("Generate Clips"):
        st.info("Processing your file... This may take a while.")

        # The output_base_dir for process_media_file will be CURRENT_RUN_OUTPUT_BASE_DIR.
        # The process_media_file function will then create a subdirectory inside this
        # based on the media file's name. E.g. CURRENT_RUN_OUTPUT_BASE_DIR/uploaded_file_basename/

        processing_successful = False
        try:
            # Call the refactored processing function
            # It's expected that process_media_file will create its own subfolder within CURRENT_RUN_OUTPUT_BASE_DIR
            # based on the input file name.
            process_media_file(temp_input_path, CURRENT_RUN_OUTPUT_BASE_DIR)
            processing_successful = True # Assume success if no exception
        except Exception as e:
            st.error(f"An error occurred during clip generation: {e}")
            # Optionally, log the full traceback here or ensure process_media_file does

        if processing_successful:
            st.success("Clip generation process completed!")

            # Construct the expected output directory path that process_media_file would have created
            base_name = os.path.splitext(uploaded_file.name)[0]
            final_output_directory = os.path.join(CURRENT_RUN_OUTPUT_BASE_DIR, base_name, "clips") # as per process_media_file structure

            if os.path.exists(final_output_directory) and os.listdir(final_output_directory):
                st.write(f"Generated clips can be found in: `{final_output_directory}`")
                st.write("You can browse this directory on the server where the app is running.")
                # Note: Streamlit doesn't directly support downloading folders.
                # Users would typically access this server path directly if running locally.
                # For deployed apps, alternative download methods (e.g., zipping and providing a download link) would be needed.
            else:
                st.warning(f"Processing seemed to complete, but the expected clips directory '{final_output_directory}' is empty or not found. The main video might have been generated without clips, or an issue occurred.")
        else:
            st.error("Clip generation failed. Check logs for details.")

st.markdown("---")
st.markdown("### Important Notes:")
st.markdown("- Ensure FFmpeg is installed and accessible in your system PATH for video processing.")
st.markdown("- Font file specified in `audio_to_youtube_shorts.py` (DEFAULT_FONT_PATH) must exist.")
st.markdown(f"- Uploaded files and generated outputs are temporarily stored in subdirectories of: `{SESSION_BASE_DIR}`")
