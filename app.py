"""
🚀 Production-Ready Streamlit Resume Screening System
Fully integrated ML/NLP pipeline - No separate backend API required
"""

import streamlit as st
import pandas as pd
import io
import sys
import os
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Add services to path
sys.path.insert(0, str(Path(__file__).parent / "ai_resume_screening" / "app"))

# Import all services
from services.pdf_parser import extract_text_from_pdf, extract_text_from_docx
from services.section_extractor import extract_sections
from services.resume_builder import build_smart_resume_text

# Import FIXED components
from services.skill_extractor_fixed import SkillExtractor
from services.semantic_matcher_fixed import SemanticMatcher
from services.role_detector_fixed import RoleDetector
from services.scorer_fixed import create_scorer

# Import ONTOLOGY components (NEW)
from services.skill_ontology import SkillOntology
from services.skill_matcher_ontology import OntologyAwareSkillMatcher
from services.scorer_upgraded import create_upgraded_scorer

# Import CONTEXT-AWARE & MULTI-RESUME components (LATEST)
from services.context_matcher import ContextAwareMatcher
from services.multi_resume_ranker import MultiResumeRanker

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="NexGen Resume Screening",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS STYLING
# ==========================================
st.markdown("""
<style>
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    .badge-excellent { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 8px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px; display: inline-block; }
    .badge-good { background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%); color: white; padding: 8px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px; display: inline-block; }
    .badge-moderate { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); color: white; padding: 8px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px; display: inline-block; }
    .badge-weak { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); color: white; padding: 8px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px; display: inline-block; }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #adb5bd;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1e2530 0%, #0f1419 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 5px solid #00f2fe;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    h1, h2, h3 { font-weight: 700; letter-spacing: -0.5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CACHE & STATE MANAGEMENT
# ==========================================
@st.cache_resource
def load_skill_extractor():
    """Cache the skill extractor"""
    return SkillExtractor()

@st.cache_resource
def load_semantic_matcher():
    """Cache the semantic matcher"""
    return SemanticMatcher()

@st.cache_resource
def load_role_detector():
    """Cache the role detector"""
    return RoleDetector()

@st.cache_resource
def load_skill_ontology():
    """Cache the skill ontology"""
    return SkillOntology()

@st.cache_resource
def load_ontology_matcher():
    """Cache the ontology-aware skill matcher"""
    return OntologyAwareSkillMatcher()

@st.cache_resource
def load_scorer():
    """Cache the UPGRADED scorer with ontology-aware matching"""
    skill_ext = load_skill_extractor()
    semantic_match = load_semantic_matcher()
    role_detect = load_role_detector()
    ontology_match = load_ontology_matcher()
    
    # Use upgraded scorer with ontology support
    return create_upgraded_scorer(
        skill_extractor=skill_ext,
        semantic_matcher=semantic_match,
        role_detector=role_detect,
        ontology_matcher=ontology_match
    )

@st.cache_resource
def load_context_matcher():
    """Cache the context-aware matcher"""
    return ContextAwareMatcher()

@st.cache_resource
def load_multi_resume_ranker():
    """Cache the multi-resume ranker"""
    scorer = load_scorer()
    context_match = load_context_matcher()
    skill_ext = load_skill_extractor()
    
    return MultiResumeRanker(
        scorer=scorer,
        context_matcher=context_match,
        skill_extractor=skill_ext
    )

# Initialize session state for caching results
if "resume_cache" not in st.session_state:
    st.session_state.resume_cache = {}

# ==========================================
# CORE PROCESSING FUNCTIONS
# ==========================================

def parse_resume(uploaded_file) -> Tuple[str, Dict[str, str]]:
    """
    Parse resume from uploaded file (PDF or DOCX) and extract sections
    """
    try:
        file_bytes = uploaded_file.getvalue()
        
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(file_bytes)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.type}")
        
        # Extract sections
        sections = extract_sections(text)
        return text, sections
        
    except Exception as e:
        st.error(f"❌ Error parsing resume: {e}")
        return None, None

def screen_resume(
    parsed_resume: Dict[str, str],
    job_role: str,
    job_description: str,
    requirements: str = ""
) -> Dict[str, Any]:
    """
    Execute full screening pipeline on parsed resume
    """
    try:
        # Build semantic text for embeddings
        smart_resume_text = build_smart_resume_text(parsed_resume)
        
        # Get scorer instance
        scorer = load_scorer()
        
        # Run the AI scorer (3-layer hybrid scoring with FIXED logic)
        result = scorer.score_candidate(
            parsed_resume=parsed_resume,
            job_role=job_role,
            requirements=requirements,
            job_description=job_description,
            resume_semantic_text=smart_resume_text
        )
        
        return result
        
    except Exception as e:
        st.error(f"❌ Scoring error: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_recommendation_color(score: float) -> str:
    """Map score to badge type"""
    s = score * 100
    if s >= 80:
        return "excellent"
    elif s >= 65:
        return "good"
    elif s >= 45:
        return "moderate"
    else:
        return "weak"

def get_recommendation_text(score: float) -> str:
    """Get recommendation tier text"""
    s = score * 100
    if s >= 80:
        return "STRONG MATCH"
    elif s >= 65:
        return "GOOD MATCH"
    elif s >= 45:
        return "MODERATE MATCH"
    else:
        return "WEAK MATCH"

# ==========================================
# UI COMPONENTS
# ==========================================

def render_header():
    """Render page header"""
    st.markdown("""
    <div class='header-container'>
        <h1>✨ NexGen Resume Screening System</h1>
        <p>AI-Powered Resume Matching with Semantic Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def render_result_card(result: Dict[str, Any], filename: str, rank: int = 0):
    """Render a beautiful result card with detailed breakdown and evidence"""
    
    final_score = result.get("final_score", 0)  # Now 0-100
    recommendation = result.get("recommendation", "UNKNOWN")
    confidence = result.get("confidence", "Unknown")
    detected_role = result.get("detected_role", "Unknown")
    
    # Badge color based on score
    if final_score >= 80:
        badge_color = "excellent"
        badge_text = "🌟 STRONG"
    elif final_score >= 65:
        badge_color = "good"
        badge_text = "✅ GOOD"
    elif final_score >= 50:
        badge_color = "moderate"
        badge_text = "⚡ MODERATE"
    else:
        badge_color = "weak"
        badge_text = "⚠️ WEAK"
    
    # Display rank if provided
    if rank > 0:
        col_rank, col_score = st.columns([1, 5])
        with col_rank:
            st.markdown(f"### Rank #{rank}")
        with col_score:
            st.markdown(f"**{filename}** - Score: **{final_score:.1f}/100**")
    
    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### 📄 {filename}")
            st.markdown(f"<span class='badge-{badge_color}'>{badge_text}</span>", unsafe_allow_html=True)
        
        with col2:
            st.metric("Final Score", f"{final_score:.1f}/100")
        
        with col3:
            st.metric("Detected Role", detected_role)
        
        st.divider()
        
        # Display context-aware evidence if available
        if result.get("strong_evidence") or result.get("evidence_summary"):
            with st.expander("📝 Evidence & Context Matching", expanded=False):
                st.markdown("**Why this candidate matches:**")
                st.info(result.get("evidence_summary", "N/A"))
                
                if result.get("strong_evidence"):
                    st.markdown("**Top Matching Sections:**")
                    for i, evidence in enumerate(result.get("strong_evidence", []), 1):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**[{i}]** {evidence.get('sentence', '')[:120]}...")
                        with col2:
                            relevance = evidence.get('relevance', 'N/A')
                            st.write(f"📊 {relevance}")
                
                # Coverage metric
                if result.get("coverage_score"):
                    coverage_val = float(result.get("coverage_score", "0%").strip("%")) / 100
                    st.progress(coverage_val, text=f"Coverage: {result.get('coverage_score')}")
            
            st.divider()
        
        # Tabs for deep analysis
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Overview", "🔧 Skills", "📊 Breakdown", "📈 Metrics"])
        
        with tab1:
            st.markdown("**Recommendation**")
            st.info(recommendation)
            st.write(f"**Confidence:** {confidence}")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric(
                    "Semantic Similarity",
                    f"{result.get('semantic_similarity_score', 0):.1f}%"
                )
            with col_b:
                st.metric(
                    "Skill Match",
                    f"{result.get('skill_match_score', 0):.1f}%"
                )
            with col_c:
                st.metric(
                    "Role Alignment",
                    f"{result.get('role_alignment_score', 0):.1f}%"
                )
        
        with tab2:
            col_match, col_miss = st.columns(2)
            
            with col_match:
                st.markdown("**✅ Matched Skills**")
                matched = result.get("matched_skills", [])
                if matched:
                    for skill in matched[:10]:
                        st.write(f"• {skill}")
                    if len(matched) > 10:
                        st.write(f"_... and {len(matched) - 10} more_")
                else:
                    st.write("None matched")
            
            with col_miss:
                st.markdown("**❌ Missing Skills**")
                missing = result.get("missing_skills", [])
                if missing:
                    for skill in missing[:10]:
                        st.write(f"• {skill}")
                    if len(missing) > 10:
                        st.write(f"_... and {len(missing) - 10} more_")
                else:
                    st.write("All required skills present! ✨")
        
        with tab3:
            st.markdown("**Detailed Score Breakdown**")
            breakdown_data = {
                "Component": ["Semantic Similarity", "Skill Matching", "Role Alignment"],
                "Score": [
                    result.get("semantic_similarity_score", 0),
                    result.get("skill_match_score", 0),
                    result.get("role_alignment_score", 0)
                ],
                "Weight": [40, 40, 20]
            }
            df_breakdown = pd.DataFrame(breakdown_data)
            df_breakdown["Contribution"] = (df_breakdown["Score"] * df_breakdown["Weight"] / 100).round(1)
            
            st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
        
        with tab4:
            st.markdown("**Section Similarities**")
            metrics_data = {
                "Section": ["Skills", "Experience", "Projects"],
                "Similarity": [
                    result.get("skills_similarity", 0),
                    result.get("experience_similarity", 0),
                    result.get("projects_similarity", 0)
                ]
            }
            df_metrics = pd.DataFrame(metrics_data)
            st.bar_chart(df_metrics.set_index("Section"))

# ==========================================
# MAIN APP LAYOUT
# ==========================================

def main():
    render_header()
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("### 📋 Job Requirements")
        
        job_role = st.text_input(
            "Job Role/Title",
            value="Senior Software Engineer",
            help="The position you're hiring for"
        )
        
        job_description = st.text_area(
            "Job Description",
            height=250,
            placeholder="Paste the full job description here (required)...",
            help="Complete job description for semantic matching"
        )
        
        requirements = st.text_area(
            "Key Requirements (Optional)",
            height=150,
            placeholder="List key requirements separated by commas or newlines...",
            help="Explicit requirements list for skill matching"
        )
        
        st.markdown("---")
        st.markdown("### 📄 Resume Upload")
    
    # Main content area
    col_main, col_sidebar = st.columns([3, 1])
    
    with col_main:
        uploaded_files = st.file_uploader(
            "Upload Resume(s)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Upload one or more resumes in PDF or DOCX format"
        )
        
        if not job_description.strip():
            st.warning("⚠️ Please enter a job description to proceed with screening.")
            return
        
        if not uploaded_files:
            st.info("👈 Upload resumes from the sidebar to get started")
            return
        
        # Process uploaded resumes using multi-resume ranker
        st.markdown("---")
        st.markdown(f"### 📊 Screening Results ({len(uploaded_files)} resume{'s' if len(uploaded_files) != 1 else ''})")
        
        # Prepare resume data for ranker
        resumes_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Parsing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}...")
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # Parse resume
            text, sections = parse_resume(uploaded_file)
            if text is None:
                st.error(f"Failed to parse {uploaded_file.name}")
                continue
            
            resumes_data.append({
                "name": uploaded_file.name,
                "text": text
            })
        
        status_text.empty()
        progress_bar.empty()
        
        if not resumes_data:
            st.error("No resumes were successfully parsed.")
            return
        
        # Use multi-resume ranker
        with st.spinner("🔍 Evaluating and ranking resumes..."):
            ranker = load_multi_resume_ranker()
            ranking_result = ranker.rank_resumes(
                resumes_data=resumes_data,
                job_role=job_role,
                requirements=requirements,
                job_description=job_description
            )
        
        results_list = ranking_result["ranked_results"]
        
        # Display ranking summary
        st.markdown("### 🏆 Ranking Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Top Candidate", ranking_result["summary"]["top_candidate"])
        with col2:
            st.metric("Top Score", f"{ranking_result['summary']['top_score']:.1f}/100")
        with col3:
            st.metric("Average Score", f"{ranking_result['summary']['average_score']:.1f}/100")
        
        st.info(ranking_result["summary"]["recommendation"])
        
        # Display results with ranking
        for result in results_list:
            render_result_card(result, result["resume_name"], rank=result.get("rank", 0))
            st.markdown("---")
        
        # Export results
        st.markdown("### 📥 Export Results")
        
        export_data = []
        for result in results_list:
            export_data.append({
                "Rank": result.get("rank", 0),
                "Filename": result.get("resume_name", "Unknown"),
                "Final Score": round(result.get("final_score", 0), 2),
                "Recommendation": result.get("recommendation", "N/A"),
                "Detected Role": result.get("detected_role", "Unknown"),
                "Semantic Score": round(result.get("semantic_score", 0), 2),
                "Skill Score": round(result.get("skill_score", 0), 2),
                "Role Alignment": round(result.get("role_score", 0), 2),
                "Matched Skills": result.get("num_skills_matched", 0),
                "Missing Skills": len(result.get("missing_skills", []))
            })
            })
        
        df_export = pd.DataFrame(export_data)
        
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📊 Download Results as CSV",
            data=csv_data,
            file_name="resume_screening_results.csv",
            mime="text/csv"
        )
        
        st.dataframe(df_export, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
