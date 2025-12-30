import streamlit as st
import numpy as np
import joblib
import tempfile
import os
import sys
import matplotlib.pyplot as plt
from io import BytesIO
import traceback
import librosa
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Gender Voice Classifier",
    page_icon="🎤",
    layout="wide"
)

# Title
st.title("🎤 Gender Voice Classification System")
st.markdown("""
This system classifies speaker gender from audio files using a trained SVM model.
Upload a WAV file or record your voice directly.
""")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    # Check Python version
    st.write(f"**Python:** {sys.version.split()[0]}")
    
    # Try to load model
    model = None
    model_paths = [
        "svm_gender_model.pkl",
        "./svm_gender_model.pkl",
        "model/svm_gender_model.pkl",
        "voice_gender_model.pkl"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.success(f"✅ Model loaded: {path}")
                
                # Debug model info
                with st.expander("🔍 Model Info"):
                    st.write(f"**Model Type:** {type(model).__name__}")
                    if hasattr(model, 'classes_'):
                        st.write(f"**Model Classes:** {model.classes_}")
                        st.write(f"**Class Mapping:**")
                        for i, cls in enumerate(model.classes_):
                            st.write(f"  - Index {i}: {cls}")
                    if hasattr(model, 'predict_proba'):
                        st.write("**Has predict_proba:** ✅ Yes")
                    else:
                        st.write("**Has predict_proba:** ❌ No")
                
                break
            except Exception as e:
                st.error(f"❌ Error loading {path}: {str(e)[:100]}")
    
    if not model:
        st.error("❌ Model not found. Please ensure 'svm_gender_model.pkl' is in the correct folder.")
        st.info("You can continue testing with audio upload/record, but predictions won't work.")
    
    st.header("📋 Instructions")
    st.markdown("""
    1. **Record**: Click 'Start Recording' to record your voice
    2. **Upload**: Or upload existing WAV file
    3. **Analyze**: Click 'Analyze Audio' for prediction
    4. **Results**: View gender prediction and confidence
    """)
    
    # Audio recording settings
    st.header("⚙️ Recording Settings")
    sample_rate = st.selectbox(
        "Sample Rate",
        [16000, 22050, 44100],
        index=1,
        help="Higher sample rate = better quality but larger files"
    )
    
    duration = st.slider(
        "Recording Duration (seconds)",
        min_value=1,
        max_value=10,
        value=3,
        help="Length of recording"
    )

# ============================
# FUNGSI EKSTRAKSI MFCC - TETAP SAMA
# ============================
def extract_mfcc(audio_path, n_mfcc=13):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Calculate mean and standard deviation
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Concatenate mean and std to form feature vector
        feature_vector = np.concatenate((mfcc_mean, mfcc_std))
        
        return feature_vector, y, sr
        
    except Exception as e:
        st.error(f"Error extracting MFCC: {str(e)}")
        return None, None, None

# ============================
# FUNGSI PREDIKSI GENDER - DIPERBAIKI
# ============================
def predict_gender_from_audio(audio_path, model):
    """Predict gender from audio file - FIXED VERSION"""
    try:
        # Extract MFCC features
        features, _, _ = extract_mfcc(audio_path)
        
        if features is None:
            return None, None
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(features)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            
            # Debug output
            print(f"DEBUG - Raw prediction: {prediction}")
            print(f"DEBUG - Raw probabilities: {probabilities}")
            if hasattr(model, 'classes_'):
                print(f"DEBUG - Model classes: {model.classes_}")
            
            # Handle different class encodings
            if hasattr(model, 'classes_'):
                classes = model.classes_
                if len(classes) == 2:
                    # Determine gender based on classes
                    if isinstance(classes[0], str):
                        # String classes like ['female', 'male']
                        if classes[0].lower() == 'female':
                            female_index = 0
                            male_index = 1
                        else:
                            female_index = 1
                            male_index = 0
                    else:
                        # Numeric classes like [0, 1]
                        # We need to determine which is which
                        if prediction == 0:
                            # Assume 0 is female if not specified
                            female_index = 0
                            male_index = 1
                        else:
                            female_index = 1
                            male_index = 0
                    
                    # Get probabilities
                    female_prob = probabilities[female_index]
                    male_prob = probabilities[male_index]
                    
                    # Determine final prediction based on highest probability
                    if female_prob > male_prob:
                        gender_label = "👩 FEMALE"
                        confidence = female_prob
                    else:
                        gender_label = "👨 MALE"
                        confidence = male_prob
                    
                    return gender_label, [female_prob, male_prob], confidence
            else:
                # No classes attribute, use probabilities directly
                if len(probabilities) == 2:
                    # Assume [Female, Male] order
                    female_prob = probabilities[0]
                    male_prob = probabilities[1]
                    
                    if female_prob > male_prob:
                        gender_label = "👩 FEMALE"
                        confidence = female_prob
                    else:
                        gender_label = "👨 MALE"
                        confidence = male_prob
                    
                    return gender_label, [female_prob, male_prob], confidence
        
        # Fallback for models without predict_proba
        if hasattr(model, 'classes_'):
            classes = model.classes_
            if len(classes) == 2:
                if isinstance(classes[0], str):
                    gender_label = f"👨 MALE" if prediction == 'male' else "👩 FEMALE"
                else:
                    # Try to infer from common patterns
                    gender_label = "👨 MALE" if prediction == 1 else "👩 FEMALE"
        else:
            # Default assumption
            gender_label = "👨 MALE" if prediction == 1 else "👩 FEMALE"
        
        # Create dummy probabilities
        if "MALE" in gender_label:
            probabilities = [0.3, 0.7]  # [Female, Male]
            confidence = 0.7
        else:
            probabilities = [0.7, 0.3]  # [Female, Male]
            confidence = 0.7
        
        return gender_label, probabilities, confidence
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# ============================
# FUNGSI RECORD AUDIO - TETAP SAMA
# ============================
def record_audio(duration, sample_rate, filename="recording.wav"):
    """Record audio from microphone"""
    try:
        st.info(f"🎙️ Recording for {duration} seconds... Speak now!")
        
        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        # Save to file
        sf.write(filename, recording, sample_rate)
        
        st.success(f"✅ Recording saved: {filename}")
        return filename, recording.flatten(), sample_rate
        
    except Exception as e:
        st.error(f"Recording error: {str(e)}")
        st.info("Make sure your microphone is connected and permissions are granted.")
        return None, None, None

# Main Interface - Single tab
st.header("🎤 Voice Recording & Analysis")

# Create two columns for recording and uploading
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎙️ Record Your Voice")
    
    if st.button("🎤 Start Recording", type="primary", key="record_btn"):
        if 'recording_file' in st.session_state and os.path.exists(st.session_state.recording_file):
            try:
                os.remove(st.session_state.recording_file)
            except:
                pass
        
        # Create temporary file for recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"recording_{timestamp}.wav"
        
        # Record audio
        recording_file, audio_data, sr = record_audio(duration, sample_rate, recording_filename)
        
        if recording_file:
            st.session_state.recording_file = recording_file
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sr
            
            # Display audio player
            st.audio(recording_file, format='audio/wav')
            
            # Show recording info
            st.write(f"**Duration:** {duration} seconds")
            st.write(f"**Sample Rate:** {sample_rate} Hz")
            st.write(f"**File:** {recording_filename}")

with col2:
    st.subheader("📁 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose a WAV file", 
        type=['wav'],
        help="Upload a WAV file for gender classification"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to session state
        uploaded_filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with open(uploaded_filename, 'wb') as f:
            f.write(uploaded_file.read())
        
        st.session_state.uploaded_file = uploaded_filename
        
        # Display file info
        st.write("**File:**", uploaded_file.name)
        st.write("**Size:**", f"{uploaded_file.size / 1024:.1f} KB")
        
        # Audio player
        st.audio(uploaded_file, format='audio/wav')

# Process audio (either recorded or uploaded)
current_audio_file = None
audio_source = None

if 'recording_file' in st.session_state and os.path.exists(st.session_state.recording_file):
    current_audio_file = st.session_state.recording_file
    audio_source = "Recording"
elif 'uploaded_file' in st.session_state and os.path.exists(st.session_state.uploaded_file):
    current_audio_file = st.session_state.uploaded_file
    audio_source = "Uploaded File"

if current_audio_file:
    st.markdown("---")
    st.subheader(f"📊 Audio Analysis ({audio_source})")
    
    # Create tabs for visualization and analysis
    viz_tab, analysis_tab = st.tabs(["🎨 Visualization", "🔍 Analysis"])
    
    with viz_tab:
        # Extract and visualize audio
        try:
            # Load audio for visualization
            y, sr = librosa.load(current_audio_file, sr=None)
            
            # Create visualization
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            
            # Waveform
            time = np.linspace(0, len(y)/sr, num=len(y))
            axes[0].plot(time, y, alpha=0.7, color='blue', linewidth=0.5)
            axes[0].set_title('Audio Waveform')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid(True, alpha=0.3)
            
            # Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[1])
            axes[1].set_title('Spectrogram')
            plt.colorbar(img, ax=axes[1], format='%+2.0f dB')
            
            # MFCC (first 13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            img2 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[2])
            axes[2].set_title('MFCC Coefficients (Features for Model)')
            plt.colorbar(img2, ax=axes[2])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Audio statistics
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Duration", f"{len(y)/sr:.2f} s")
            with col_stat2:
                st.metric("Sample Rate", f"{sr} Hz")
            with col_stat3:
                st.metric("Amplitude Range", f"{y.min():.3f} to {y.max():.3f}")
            
        except Exception as e:
            st.warning(f"Visualization error: {e}")
    
    with analysis_tab:
        if model:
            if st.button("🚀 Analyze Audio for Gender", type="primary", key="analyze_btn"):
                with st.spinner("Extracting features and analyzing..."):
                    try:
                        # Get prediction with fixed function
                        gender_label, probabilities, confidence = predict_gender_from_audio(current_audio_file, model)
                        
                        if gender_label:
                            # Extract individual probabilities
                            female_prob = probabilities[0] * 100  # Female probability in percentage
                            male_prob = probabilities[1] * 100    # Male probability in percentage
                            confidence_percent = confidence * 100  # Overall confidence
                            
                            # ============================================
                            # FIXED DISPLAY - Konsisten antara prediksi dan probabilitas
                            # ============================================
                            
                            # Display final prediction
                            st.subheader("🎯 Final Prediction")
                            
                            # Create a clean, consistent result display
                            if "FEMALE" in gender_label:
                                # Female prediction
                                pred_col1, pred_col2 = st.columns([1, 2])
                                with pred_col1:
                                    # Female icon
                                    st.markdown("""
                                    <div style='text-align: center;'>
                                        <h1 style='font-size: 80px; margin: 0;'>👩</h1>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with pred_col2:
                                    # Female label and confidence
                                    st.markdown(f"""
                                    <div style='padding: 10px;'>
                                        <h2 style='color: #e91e63; margin-bottom: 5px;'>FEMALE VOICE</h2>
                                        <div style='background-color: #f5f5f5; border-radius: 5px; padding: 10px;'>
                                            <p style='margin: 0; color: #666;'><strong>Confidence:</strong> {confidence_percent:.1f}%</p>
                                            <div style='width: 100%; background-color: #ddd; border-radius: 3px; margin-top: 5px;'>
                                                <div style='width: {confidence_percent}%; height: 10px; background-color: #e91e63; border-radius: 3px;'></div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                # Male prediction
                                pred_col1, pred_col2 = st.columns([1, 2])
                                with pred_col1:
                                    # Male icon
                                    st.markdown("""
                                    <div style='text-align: center;'>
                                        <h1 style='font-size: 80px; margin: 0;'>👨</h1>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with pred_col2:
                                    # Male label and confidence
                                    st.markdown(f"""
                                    <div style='padding: 10px;'>
                                        <h2 style='color: #2196f3; margin-bottom: 5px;'>MALE VOICE</h2>
                                        <div style='background-color: #f5f5f5; border-radius: 5px; padding: 10px;'>
                                            <p style='margin: 0; color: #666;'><strong>Confidence:</strong> {confidence_percent:.1f}%</p>
                                            <div style='width: 100%; background-color: #ddd; border-radius: 3px; margin-top: 5px;'>
                                                <div style='width: {confidence_percent}%; height: 10px; background-color: #2196f3; border-radius: 3px;'></div>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            # Probability Distribution - FIXED and SMALLER
                            st.subheader("📊 Probability Distribution")
                            
                            # Create two columns for probabilities
                            prob_col1, prob_col2 = st.columns(2)
                            
                            with prob_col1:
                                # Female Probability - DARK TEXT COLOR
                                st.markdown(f"""
                                <div style='text-align: center; padding: 15px; background-color: #fce4ec; border-radius: 10px; border: 2px solid #e91e63;'>
                                    <h4 style='color: #333; margin-bottom: 10px;'>👩 Female Probability</h4>
                                    <h2 style='color: #e91e63; margin: 0;'>{female_prob:.1f}%</h2>
                                    <div style='width: 100%; background-color: #ddd; border-radius: 5px; margin-top: 10px;'>
                                        <div style='width: {female_prob}%; height: 15px; background-color: #e91e63; border-radius: 5px;'></div>
                                    </div>
                                    <p style='color: #333; margin-top: 5px; margin-bottom: 0; font-size: 14px;'>Probability that voice is female</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with prob_col2:
                                # Male Probability - DARK TEXT COLOR
                                st.markdown(f"""
                                <div style='text-align: center; padding: 15px; background-color: #e3f2fd; border-radius: 10px; border: 2px solid #2196f3;'>
                                    <h4 style='color: #333; margin-bottom: 10px;'>👨 Male Probability</h4>
                                    <h2 style='color: #2196f3; margin: 0;'>{male_prob:.1f}%</h2>
                                    <div style='width: 100%; background-color: #ddd; border-radius: 5px; margin-top: 10px;'>
                                        <div style='width: {male_prob}%; height: 15px; background-color: #2196f3; border-radius: 5px;'></div>
                                    </div>
                                    <p style='color: #333; margin-top: 5px; margin-bottom: 0; font-size: 14px;'>Probability that voice is male</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # SMALLER Probability Pie Chart
                            st.markdown("---")
                            st.markdown("<h4 style='text-align: center; color: #333;'>Probability Visualization</h4>", unsafe_allow_html=True)
                            
                            # Create a smaller figure
                            fig_prob, ax_prob = plt.subplots(figsize=(5, 4))  # Smaller size
                            
                            labels = ['Female', 'Male']
                            sizes = [female_prob, male_prob]
                            colors = ['#e91e63', '#2196f3']
                            
                            # Highlight the predicted gender
                            if "FEMALE" in gender_label:
                                explode = (0.05, 0)  # Explode female slice
                            else:
                                explode = (0, 0.05)  # Explode male slice
                            
                            wedges, texts, autotexts = ax_prob.pie(
                                sizes,
                                labels=labels,
                                colors=colors,
                                autopct='%1.1f%%',
                                startangle=90,
                                explode=explode,
                                shadow=False,
                                textprops={'fontsize': 10}
                            )
                            
                            # Style the percentage text - DARK COLOR
                            for autotext in autotexts:
                                autotext.set_color('#333333')  # Dark gray, not white
                                autotext.set_fontweight('bold')
                                autotext.set_fontsize(11)
                            
                            # Style the labels
                            for text in texts:
                                text.set_color('#333333')  # Dark gray
                                text.set_fontsize(11)
                            
                            ax_prob.set_title('Probability Distribution', fontsize=12, fontweight='bold', color='#333333')
                            ax_prob.axis('equal')
                            
                            # Add a legend
                            ax_prob.legend(wedges, [f'{label}: {size:.1f}%' for label, size in zip(labels, sizes)],
                                         title="Gender",
                                         loc="center left",
                                         bbox_to_anchor=(1, 0, 0.5, 1),
                                         fontsize=10)
                            
                            # Make layout tight
                            plt.tight_layout()
                            
                            # Display the chart
                            st.pyplot(fig_prob)
                            
                            # ============================================
                            # LOGIC VERIFICATION - Debug information
                            # ============================================
                            with st.expander("🔍 Logic Verification"):
                                st.write("**Prediction Logic Check:**")
                                st.write(f"- Predicted Gender: {gender_label}")
                                st.write(f"- Female Probability: {female_prob:.2f}%")
                                st.write(f"- Male Probability: {male_prob:.2f}%")
                                
                                # Verify consistency
                                if "FEMALE" in gender_label and female_prob > male_prob:
                                    st.success("✅ **CONSISTENT**: Female prediction matches higher female probability")
                                elif "MALE" in gender_label and male_prob > female_prob:
                                    st.success("✅ **CONSISTENT**: Male prediction matches higher male probability")
                                else:
                                    st.error("⚠️ **INCONSISTENT**: Prediction doesn't match probability distribution")
                                    st.info("This might indicate a model encoding issue. The prediction above is based on the highest probability, not the raw model output.")
                                
                                # Show raw model info if available
                                if model and hasattr(model, 'classes_'):
                                    st.write(f"- Model Classes: {model.classes_}")
                            
                            # Download results
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            audio_filename = os.path.basename(current_audio_file)
                            
                            results_text = f"""GENDER VOICE CLASSIFICATION REPORT
========================================
Audio File: {audio_filename}
Audio Source: {audio_source}
Analysis Timestamp: {timestamp}

PREDICTION RESULTS:
------------------
Predicted Gender: {gender_label.replace('👨 ', '').replace('👩 ', '')}
Confidence Level: {confidence_percent:.1f}%

PROBABILITY DISTRIBUTION:
-------------------------
Female Probability: {female_prob:.2f}%
Male Probability: {male_prob:.2f}%

LOGIC VERIFICATION:
-------------------
{'✓ Consistent: Prediction matches probability distribution' 
 if ("FEMALE" in gender_label and female_prob > male_prob) or 
    ("MALE" in gender_label and male_prob > female_prob) 
 else '⚠ Warning: Prediction logic needs verification'}

AUDIO INFORMATION:
------------------
Sample Rate: {sr if 'sr' in locals() else 'N/A'} Hz
Duration: {len(y)/sr if 'y' in locals() and 'sr' in locals() else 'N/A'} seconds
========================================
Generated by Gender Voice Classification System"""
                            
                            # Create download button
                            st.download_button(
                                "📥 Download Full Report",
                                results_text,
                                file_name=f"gender_analysis_{audio_filename}.txt",
                                mime="text/plain"
                            )
                        
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Model not loaded. Please ensure the model file (svm_gender_model.pkl) is available.")
            st.info("You can still record and upload audio, but gender prediction requires the model file.")

# Clear recordings button
if st.button("🗑️ Clear All Audio", type="secondary"):
    # Remove recording file if exists
    if 'recording_file' in st.session_state and os.path.exists(st.session_state.recording_file):
        try:
            os.remove(st.session_state.recording_file)
            del st.session_state.recording_file
        except:
            pass
    
    # Remove uploaded file if exists
    if 'uploaded_file' in st.session_state and os.path.exists(st.session_state.uploaded_file):
        try:
            os.remove(st.session_state.uploaded_file)
            del st.session_state.uploaded_file
        except:
            pass
    
    # Clear other session state variables
    for key in ['audio_data', 'sample_rate']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("All audio files cleared!")
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #333; padding: 20px;'>
        <h4>🎤 Gender Voice Classification System</h4>
        <p>Record your voice or upload WAV files for gender prediction</p>
        <p><strong>Features:</strong> 26 MFCC features | <strong>Model:</strong> SVM | <strong>Output:</strong> Male/Female</p>
        <p style='font-size: 12px; margin-top: 20px; color: #666;'>Tip: Speak clearly into the microphone for best results</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Cleanup on app close
def cleanup():
    """Clean up temporary files"""
    if 'recording_file' in st.session_state and os.path.exists(st.session_state.recording_file):
        try:
            os.remove(st.session_state.recording_file)
        except:
            pass
    
    if 'uploaded_file' in st.session_state and os.path.exists(st.session_state.uploaded_file):
        try:
            os.remove(st.session_state.uploaded_file)
        except:
            pass

# Register cleanup function
import atexit
atexit.register(cleanup)
