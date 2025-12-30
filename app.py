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
Upload a WAV file or use the test interface.
""")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    # Check Python version
    st.write(f"**Python:** {sys.version.split()[0]}")
    
    # Try to load model
    model = None
    model_paths = [
        "/content/drive/MyDrive/Pengolahan Bunyi Digital /svm_gender_model.pkl",
        "svm_gender_model.pkl",
        "./svm_gender_model.pkl",
        "model/svm_gender_model.pkl"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                st.success(f"✅ Model loaded: {path}")
                break
            except Exception as e:
                st.error(f"❌ Error loading {path}: {str(e)[:100]}")
    
    if not model:
        st.error("❌ Model not found. Please ensure 'svm_gender_model.pkl' is in the correct folder.")
    
    st.header("📋 Instructions")
    st.markdown("""
    1. Upload WAV audio file
    2. View audio visualization
    3. Get gender prediction
    4. Download results
    """)
    
    # Add debug info
    with st.expander("🔧 Debug Info"):
        st.write("**Working Directory:**", os.getcwd())
        st.write("**Files in directory:**", os.listdir('.'))
        st.write("**Librosa Version:**", librosa.__version__ if 'librosa' in sys.modules else "Not installed")

# ============================
# FUNGSI EKSTRAKSI MFCC
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
# FUNGSI PREDIKSI GENDER
# ============================
def predict_gender_from_audio(audio_path, model):
    """Predict gender from audio file"""
    try:
        # Extract MFCC features
        features, _, _ = extract_mfcc(audio_path)
        
        if features is None:
            return None, None
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
        else:
            # If model doesn't have predict_proba, create dummy probabilities
            if prediction == 0:
                probabilities = [0.8, 0.2]  # Male
            else:
                probabilities = [0.2, 0.8]  # Female
        
        # Map prediction to label (sesuai dengan model Anda)
        # 0 = Female, 1 = Male (sesuai kode Anda)
        gender_label = "👨 MALE" if prediction == 1 else "👩 FEMALE"
        
        return gender_label, probabilities
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Main Interface
tab1, tab2, tab3 = st.tabs(["🎵 Upload Audio", "🧪 Manual Test", "📊 Model Info"])

with tab1:
    st.header("Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose a WAV file", 
        type=['wav', 'mp3', 'flac'],
        help="Upload an audio file for gender classification"
    )
    
    if uploaded_file is not None:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Display file info
        col1, col2 = st.columns(2)
        with col1:
            st.write("**File:**", uploaded_file.name)
            st.write("**Size:**", f"{uploaded_file.size / 1024:.1f} KB")
        
        # Audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Extract and visualize audio
        try:
            # Load audio for visualization
            y, sr = librosa.load(tmp_path, sr=None)
            
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
            axes[2].set_title('MFCC Coefficients')
            plt.colorbar(img2, ax=axes[2])
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Visualization error: {e}")
            # Create simple plot as fallback
            fig, ax = plt.subplots(figsize=(10, 3))
            audio_bytes = uploaded_file.getvalue()
            byte_data = np.frombuffer(audio_bytes[:10000], dtype=np.uint8)
            ax.plot(byte_data[:1000], alpha=0.7, color='blue')
            ax.set_title('Audio Data Visualization')
            ax.set_xlabel('Byte Position')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Make prediction
        if model and st.button("🚀 Analyze Audio", type="primary"):
            with st.spinner("Extracting features and analyzing..."):
                try:
                    gender_label, probabilities = predict_gender_from_audio(tmp_path, model)
                    
                    if gender_label:
                        # Display results
                        st.subheader("🎯 Analysis Results")
                        
                        result_col1, result_col2 = st.columns(2)
                        
                        with result_col1:
                            st.metric("Prediction", gender_label)
                        
                        with result_col2:
                            confidence = max(probabilities) * 100
                            st.metric("Confidence", f"{confidence:.1f}%")
                        
                        # Detailed probabilities
                        with st.expander("📈 Detailed Probabilities"):
                            col_prob1, col_prob2 = st.columns(2)
                            with col_prob1:
                                st.progress(probabilities[0], text=f"Male: {probabilities[0]*100:.1f}%")
                            with col_prob2:
                                st.progress(probabilities[1], text=f"Female: {probabilities[1]*100:.1f}%")
                        
                        # Download results
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        results_text = f"""Gender Classification Report
File: {uploaded_file.name}
Prediction: {gender_label}
Confidence: {confidence:.1f}%
Male Probability: {probabilities[0]*100:.2f}%
Female Probability: {probabilities[1]*100:.2f}%
Timestamp: {timestamp}"""
                        
                        st.download_button(
                            "📥 Download Report",
                            results_text,
                            file_name=f"gender_report_{uploaded_file.name}.txt"
                        )
                    
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Clean up
        os.unlink(tmp_path)

with tab2:
    st.header("Manual Feature Testing")
    st.markdown("Enter the 26 MFCC feature values manually:")
    
    if not model:
        st.error("Model not loaded. Cannot make predictions.")
    else:
        # Create input fields in a grid
        features = []
        
        # 4 columns for 26 features
        cols = st.columns(4)
        for i in range(26):
            with cols[i % 4]:
                features.append(
                    st.number_input(
                        f"MFCC {i//2 + 1} ({'Mean' if i < 13 else 'Std'})",
                        value=0.0,
                        step=0.1,
                        format="%.3f",
                        key=f"feat_{i}",
                        help=f"Feature {i+1}: {'Mean' if i < 13 else 'Standard Deviation'} of MFCC coefficient {i%13 + 1}"
                    )
                )
        
        if st.button("🔍 Predict Gender", type="primary"):
            try:
                X = np.array(features).reshape(1, -1)
                prediction = model.predict(X)[0]
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)[0]
                else:
                    probabilities = [0.8, 0.2] if prediction == 1 else [0.2, 0.8]
                
                # Map prediction to label
                gender_label = "👨 MALE" if prediction == 1 else "👩 FEMALE"
                
                # Display results creatively
                st.subheader("📊 Prediction Results")
                
                # Use columns for layout
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    if "MALE" in gender_label:
                        st.markdown("""
                        <div style='text-align: center; padding: 20px; background-color: #e3f2fd; border-radius: 10px;'>
                            <h1>👨</h1>
                            <h3>MALE</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='text-align: center; padding: 20px; background-color: #fce4ec; border-radius: 10px;'>
                            <h1>👩</h1>
                            <h3>FEMALE</h3>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_res2:
                    # Male probability gauge
                    male_pct = probabilities[1] * 100  # Index 1 is Male in your model
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <h4>Male Probability</h4>
                        <h2 style='color: #2196f3;'>{male_pct:.1f}%</h2>
                        <progress value="{male_pct}" max="100" style="width: 100%; height: 20px;"></progress>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res3:
                    # Female probability gauge
                    female_pct = probabilities[0] * 100  # Index 0 is Female in your model
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <h4>Female Probability</h4>
                        <h2 style='color: #e91e63;'>{female_pct:.1f}%</h2>
                        <progress value="{female_pct}" max="100" style="width: 100%; height: 20px;"></progress>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Feature visualization
                st.subheader("📈 Feature Visualization")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Bar chart of features
                colors = ['blue' if i < 13 else 'red' for i in range(26)]
                labels = [f"M{i+1}" for i in range(13)] + [f"S{i+1}" for i in range(13)]
                ax1.bar(range(26), features, alpha=0.7, color=colors)
                ax1.set_title('MFCC Feature Values (M=Mean, S=Std)')
                ax1.set_xlabel('Feature Index')
                ax1.set_ylabel('Value')
                ax1.set_xticks(range(26))
                ax1.set_xticklabels(labels, rotation=45, ha='right')
                ax1.grid(True, alpha=0.3)
                
                # Probability pie chart
                labels_pie = ['Female', 'Male']  # Sesuai dengan model Anda
                sizes = probabilities * 100
                colors_pie = ['#e91e63', '#2196f3']
                ax2.pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Gender Probability Distribution')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

with tab3:
    st.header("Model Information")
    
    if model:
        try:
            # Get model parameters
            st.subheader("📋 Model Details")
            st.write(f"**Model Type:** {type(model).__name__}")
            
            # Check if it's a pipeline
            if hasattr(model, 'named_steps'):
                st.write("**Pipeline Steps:**")
                for name, step in model.named_steps.items():
                    st.write(f"  - {name}: {type(step).__name__}")
            
            # Model parameters
            if hasattr(model, 'get_params'):
                with st.expander("🔧 Model Parameters"):
                    params = model.get_params()
                    for key, value in list(params.items())[:10]:  # Show first 10 params
                        st.write(f"**{key}:** {value}")
            
            # Feature information
            st.subheader("📊 Feature Information")
            st.write("**Number of features:** 26")
            st.write("**MFCC Coefficients:** 13")
            st.write("**Statistics per coefficient:** Mean (13), Standard Deviation (13)")
            
            # Class mapping
            st.subheader("🎯 Class Mapping")
            st.write("**0:** 👩 FEMALE")
            st.write("**1:** 👨 MALE")
            
            # Expected performance
            st.subheader("📈 Expected Performance")
            st.info("""
            **Training Configuration:**
            - Features: 13 MFCC coefficients with mean & std
            - Model: Support Vector Machine (SVM)
            - Total features: 26
            
            **Sample prediction mapping:**
            - Female voice → Prediction: 0
            - Male voice → Prediction: 1
            """)
            
        except Exception as e:
            st.warning(f"Limited model info: {e}")
    else:
        st.error("No model loaded")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Gender Voice Classification | SVM Model | Streamlit App</p>
        <p>Features: 26 MFCC features (13 means + 13 standard deviations)</p>
        <p>Model: SVM | Classes: Female (0), Male (1)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if 'timestamp' not in st.session_state:
    from datetime import datetime
    st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")