"""
Main Plant Disease Detection & AI Agronomy Advisory Application
Integrates PyTorch Vision Model with RAG Knowledge Retrieval.
Run with: streamlit run app.py  (or: python app.py)
"""

import os
import sys
from pathlib import Path
from PIL import Image
import torch
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from train import CLASSES, prepare_starter_data, DATA_DIR, synthesize_leaf
from predict import get_predictor
from rag import get_rag

st.set_page_config(
    page_title="PlantAI Vision - NVIDIA DGX",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design Styling (Dark theme with NVIDIA green & Emerald accents)
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f14;
        color: #f3f4f6;
    }
    .metric-card {
        background: rgba(17, 24, 34, 0.85);
        border: 1px solid rgba(118, 185, 0, 0.3);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-dgx {
        background: #76b900;
        color: #000;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 800;
        font-size: 11px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        color: #76b900 !important;
        border-bottom-color: #76b900 !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Sidebar Hardware & System Info
    st.sidebar.title("🌿 PlantAI System")
    st.sidebar.markdown("**NVIDIA DGX AI Platform**")

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        st.sidebar.success(f"🚀 **CUDA Active**\n\n{device_name} ({vram} GB VRAM)")
        st.sidebar.caption("⚡ Tensor Core Mixed Precision (AMP) Enabled")
    else:
        st.sidebar.warning("💻 **CPU Mode** (CUDA offline or workstation development)")

    # Cloud & Graph Integration Status
    nv_key = os.getenv("NVIDIA_API_KEY")
    if nv_key and nv_key.startswith("nvapi-"):
        st.sidebar.info("🔑 **NVIDIA NIM / NGC API Configured**")

    try:
        from neo4j_graph import get_neo4j_graph
        graph = get_neo4j_graph()
        if graph.is_connected():
            st.sidebar.success("🌐 **Neo4j GraphRAG Connected**")
        else:
            st.sidebar.caption("🌐 Neo4j GraphRAG: Standby (Vector RAG active)")
    except Exception:
        pass

    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")
    app_mode = st.sidebar.radio("Select View:", [
        "🔍 Leaf Disease Diagnosis & RAG",
        "💬 AI Agronomist Q&A (RAG)",
        "🤗 Hugging Face Datasets Hub",
        "📚 Knowledge Base Manuals",
        "⚙️ System & Model Info"
    ])

    # Ensure starter data and sample leaves exist
    samples_dir = BASE_DIR / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    if not any(samples_dir.glob("*.jpg")):
        for c in CLASSES:
            leaf = synthesize_leaf(c, size=256)
            leaf.save(samples_dir / f"{c}.jpg")

    predictor = get_predictor()
    rag = get_rag()

    # -------------------------------------------------------------
    # 1. Leaf Disease Diagnosis & RAG
    # -------------------------------------------------------------
    if app_mode == "🔍 Leaf Disease Diagnosis & RAG":
        st.title("🌿 Neural Plant Disease Detection & AI Agronomy Advisory")
        st.caption("Computer Vision Classification + RAG Document Retrieval for Precision Agriculture")

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("1. Input Leaf Image")
            input_type = st.radio("Choose Input Method:", ["Upload Leaf Photo", "Select from Sample Leaf Gallery"], horizontal=True)

            selected_image = None
            if input_type == "Upload Leaf Photo":
                file = st.file_uploader("Upload leaf image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
                if file:
                    try:
                        selected_image = Image.open(file).convert("RGB")
                        st.image(selected_image, caption="Uploaded Leaf", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading uploaded image: {e}")
            else:
                # Ensure samples exist
                sample_files = sorted(list(samples_dir.glob("*.jpg")))
                if not sample_files:
                    with st.spinner("Generating sample leaves..."):
                        for c in CLASSES:
                            leaf = synthesize_leaf(c, size=256)
                            leaf.save(samples_dir / f"{c}.jpg")
                        sample_files = sorted(list(samples_dir.glob("*.jpg")))

                if sample_files:
                    sample_map = {f.stem.replace("___", " - ").replace("_", " "): f for f in sample_files}
                    chosen_name = st.selectbox("Choose sample leaf:", list(sample_map.keys()))
                    if chosen_name:
                        chosen_path = sample_map[chosen_name]
                        try:
                            selected_image = Image.open(chosen_path).convert("RGB")
                            st.image(selected_image, caption=f"Sample: {chosen_name}", use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading sample image: {e}")
                else:
                    st.warning("No sample images available. Please upload a leaf photo.")

        with col2:
            st.subheader("2. Diagnosis & Agronomy Report")

            if selected_image is not None:
                if st.button("🚀 Run Neural Diagnosis", type="primary", use_container_width=True):
                    with st.spinner("Analyzing foliage patterns with PyTorch model..."):
                        result = predictor.predict(selected_image)

                        primary = result["primary"]
                        rag_profile = result["rag_knowledge"]

                        # Metrics row
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Crop", primary["crop"])
                        m2.metric("Confidence", f"{primary['confidence']}%")
                        m3.metric("Latency", f"{result['latency_ms']} ms")

                        # Diagnosis Banner
                        st.markdown(f"### Diagnosis: **{primary['crop']} {primary['disease']}**")

                        # Top Differential Candidates
                        st.markdown("**Top Differential Predictions:**")
                        for cand in result["top_candidates"]:
                            st.write(f"• **{cand['crop']} - {cand['disease']}**: {cand['confidence']}%")
                            st.progress(float(cand["confidence"]) / 100.0)

                        st.markdown("---")
                        st.markdown("### 📖 Verified Agronomy Treatment (RAG Retrieved)")

                        tab_sym, tab_org, tab_chem, tab_prev = st.tabs([
                            "🔍 Symptoms",
                            "🌱 Organic Care",
                            "🧪 Chemical Controls",
                            "🛡️ Prevention"
                        ])

                        with tab_sym:
                            symptoms = rag_profile.get("symptoms", [])
                            if symptoms:
                                for s in symptoms:
                                    st.markdown(f"- {s}")
                            else:
                                st.info("No specific symptom markers recorded.")

                        with tab_org:
                            orgs = rag_profile.get("organic_treatments", [])
                            if orgs:
                                for o in orgs:
                                    st.markdown(f"- {o}")
                            else:
                                st.info("No organic interventions listed.")

                        with tab_chem:
                            chems = rag_profile.get("chemical_treatments", [])
                            if chems:
                                for c in chems:
                                    st.markdown(f"- {c}")
                            else:
                                st.info("No chemical fungicides required.")

                        with tab_prev:
                            prevs = rag_profile.get("prevention", [])
                            if prevs:
                                for p in prevs:
                                    st.markdown(f"- {p}")
                            else:
                                st.info("Follow general field sanitation and drip irrigation.")
            else:
                st.info("👈 Please upload or select a plant leaf image to begin diagnosis.")

    # -------------------------------------------------------------
    # 2. AI Agronomist Q&A (RAG Chat)
    # -------------------------------------------------------------
    elif app_mode == "💬 AI Agronomist Q&A (RAG)":
        st.title("💬 AI Agronomist Q&A")
        st.caption("Ask specific questions regarding plant diseases, dosages, fungicides, and crop care.")

        crop_filter = st.selectbox("Target Crop Context:", ["All Crops", "Tomato", "Potato", "Rice"])
        selected_crop = None if crop_filter == "All Crops" else crop_filter

        user_query = st.text_input(
            "Ask a question to the Agronomy Knowledge Base:",
            placeholder="e.g. What fungicide dosage should I apply for late blight in potatoes?"
        )

        col_presets = st.columns(3)
        if col_presets[0].button("💡 How to treat early blight organically?"):
            user_query = "How to treat early blight organically?"
        if col_presets[1].button("💡 What are the symptoms of rice blast?"):
            user_query = "What are the symptoms of rice blast?"
        if col_presets[2].button("💡 How to manage late blight before rain?"):
            user_query = "How to manage late blight before rain?"

        if user_query:
            with st.spinner("Searching indexed agronomy documentation..."):
                answer_data = rag.answer_query(user_query, crop=selected_crop)

                st.markdown("### 📋 Agronomist Recommendation:")
                st.markdown(answer_data["answer"])

                if answer_data.get("sources"):
                    st.caption(f"📚 Grounded in reference documents: {', '.join(answer_data['sources'])}")

                with st.expander("View Retrieved Document Excerpts"):
                    st.text(answer_data.get("context", "No direct excerpts."))

    # -------------------------------------------------------------
    # 3. Hugging Face Datasets Hub
    # -------------------------------------------------------------
    elif app_mode == "🤗 Hugging Face Datasets Hub":
        st.title("🤗 Hugging Face Plant Pathology Datasets")
        st.caption("Stream, download, and prepare agricultural computer vision datasets directly from Hugging Face Hub.")

        from hf_dataset import RECOMMENDED_HF_DATASETS, download_hf_dataset

        st.subheader("1. Select or Enter Hugging Face Dataset")
        preset_names = list(RECOMMENDED_HF_DATASETS.keys())
        preset_selection = st.selectbox(
            "Recommended Datasets on Hugging Face:",
            preset_names,
            format_func=lambda k: f"{k} ({RECOMMENDED_HF_DATASETS[k]})"
        )

        use_custom = st.checkbox("Enter a custom Hugging Face dataset repository ID")
        if use_custom:
            target_repo = st.text_input("Hugging Face Dataset Repo (e.g. username/dataset-name):", value="faisal-hugging-face/plant-disease")
        else:
            target_repo = preset_selection

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            max_samples = st.selectbox(
                "Samples per class (Limit for faster training):",
                [25, 50, 100, 250, 500, "All Available"],
                index=1
            )
            limit_val = None if max_samples == "All Available" else int(max_samples)

        with col_cfg2:
            dest_dir = st.text_input("Destination Folder:", value="data/plantvillage")

        if st.button("📥 Download & Prepare Dataset from Hugging Face", type="primary", use_container_width=True):
            with st.spinner(f"Downloading and structuring '{target_repo}' from Hugging Face Hub..."):
                try:
                    res = download_hf_dataset(
                        repo_id=target_repo,
                        output_dir=BASE_DIR / dest_dir,
                        max_samples_per_class=limit_val
                    )
                    st.success(f"✅ Successfully downloaded dataset from Hugging Face into `{dest_dir}`!")
                    if "train" in res:
                        st.info(f"📊 Exported **{res['train']}** training samples and **{res['val']}** validation samples.")
                except Exception as e:
                    st.error(f"Error downloading dataset: {e}")

        # Current Local Dataset Overview
        st.markdown("---")
        st.subheader("2. Current Local Dataset Inspection")
        train_path = BASE_DIR / dest_dir / "train"
        if train_path.exists():
            class_folders = [f.name for f in train_path.iterdir() if f.is_dir()]
            st.write(f"**Total Local Classes Found:** `{len(class_folders)}`")
            class_stats = []
            for cf in sorted(class_folders):
                n_imgs = len(list((train_path / cf).glob("*.jpg"))) + len(list((train_path / cf).glob("*.png")))
                class_stats.append({"Class Name": cf, "Training Images": n_imgs})
            st.dataframe(class_stats, use_container_width=True)
        else:
            st.info(f"Directory `{dest_dir}/train` is empty or not yet created.")

    # -------------------------------------------------------------
    # 4. Knowledge Base Manuals
    # -------------------------------------------------------------
    elif app_mode == "📚 Knowledge Base Manuals":
        st.title("📚 Agronomy Knowledge Manuals")
        st.caption("Browse original reference manuals indexed in documents/")

        doc_choice = st.selectbox("Select Crop Manual:", ["Tomato Manual", "Potato Manual", "Rice Manual"])
        file_map = {
            "Tomato Manual": BASE_DIR / "documents" / "tomato.txt",
            "Potato Manual": BASE_DIR / "documents" / "potato.txt",
            "Rice Manual": BASE_DIR / "documents" / "rice.txt"
        }

        target_file = file_map[doc_choice]
        if target_file.exists():
            text = target_file.read_text(encoding="utf-8")
            st.text_area("Document Content", text, height=550)
            st.download_button(f"📥 Download {target_file.name}", text, file_name=target_file.name)
        else:
            st.error(f"File {target_file} not found.")

    # -------------------------------------------------------------
    # 4. System & Model Info
    # -------------------------------------------------------------
    elif app_mode == "⚙️ System & Model Info":
        st.title("⚙️ NVIDIA DGX System & Model Telemetry")

        st.markdown("### Hardware Overview")
        hw_col1, hw_col2 = st.columns(2)
        with hw_col1:
            st.write(f"**CUDA Available:** `{cuda_ok}`")
            if cuda_ok:
                st.write(f"**Device Name:** `{torch.cuda.get_device_name(0)}`")
                st.write(f"**VRAM Total:** `{vram} GB`")
                st.write(f"**CUDA Version:** `{torch.version.cuda}`")
            else:
                st.write("**Execution Engine:** CPU")

        with hw_col2:
            st.write(f"**PyTorch Version:** `{torch.__version__}`")
            st.write(f"**Total Model Classes:** `{len(CLASSES)}`")
            st.write(f"**Indexed RAG Chunks:** `{len(rag.chunks)}`")

        st.markdown("### Supported Disease Classes")
        st.table([{"Class Index": i, "Category": c.replace('___', ' : ').replace('_', ' ')} for i, c in enumerate(CLASSES)])


if __name__ == "__main__":
    # If run via `python app.py`, launch Streamlit automatically
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        main()
    else:
        # Check if already running inside Streamlit
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                main()
            else:
                import subprocess
                print("[App] Launching Streamlit web interface on http://localhost:8501 ...")
                subprocess.run(["streamlit", "run", str(Path(__file__).resolve()), "--server.port=8501", "--server.address=0.0.0.0"])
        except Exception:
            main()
