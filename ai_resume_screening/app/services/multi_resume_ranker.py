"""
MULTI-RESUME RANKING: ATS-Style Resume Ranking
- Scores multiple resumes against a job description
- Ranks resumes by score
- Returns structured output with evidence for each resume
- Supports batch processing
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class MultiResumeRanker:
    """
    Ranks multiple resumes against a job description
    """
    
    def __init__(self, scorer, context_matcher, skill_extractor):
        """
        Initialize ranker with scoring components
        
        Args:
            scorer: UpgradedAIScorer instance
            context_matcher: ContextAwareMatcher instance
            skill_extractor: SkillExtractor instance
        """
        self.scorer = scorer
        self.context_matcher = context_matcher
        self.skill_extractor = skill_extractor
    
    def score_single_resume(
        self,
        resume_text: str,
        resume_name: str,
        job_role: str,
        requirements: str,
        job_description: str
    ) -> Dict[str, Any]:
        """
        Score a single resume and extract evidence
        
        Returns comprehensive scoring result with context awareness
        """
        
        # Get AI score (existing logic)
        parsed_resume = {
            "summary": resume_text[:500],
            "skills": resume_text,
            "experience": resume_text,
            "projects": resume_text,
            "education": resume_text,
            "certifications": "",
            "full_text": resume_text
        }
        
        ai_score = self.scorer.score_candidate(
            parsed_resume=parsed_resume,
            job_role=job_role,
            requirements=requirements,
            job_description=job_description,
            resume_semantic_text=resume_text
        )
        
        # Get context-aware evidence
        evidence = self.context_matcher.extract_evidence(
            resume_text=resume_text,
            job_description=job_description,
            num_evidence=3
        )
        
        # Extract skills
        extracted_skills = self.skill_extractor.extract_skills(resume_text)
        required_skills = self.skill_extractor.extract_from_requirements(requirements)
        
        # Compile result
        result = {
            "resume_name": resume_name,
            "final_score": ai_score["final_score"],
            "recommendation": ai_score["recommendation"],
            "confidence": ai_score["confidence"],
            "rank": 0,  # Will be assigned after sorting
            
            # Skills information
            "matched_skills": ai_score["exact_matches"],
            "partial_match_skills": [
                f"{m['required']} (has {m['candidate']})"
                for m in ai_score["partial_matches"]
            ] if "partial_matches" in ai_score else [],
            "missing_skills": ai_score["missing_skills"],
            
            # Score breakdown explicitly mapping what frontend expects
            "semantic_similarity_score": ai_score.get("semantic_similarity_score", 0),
            "skill_match_score": ai_score.get("skill_match_score", 0),
            "role_alignment_score": ai_score.get("role_alignment_score", 0),
            
            # Legacy fields for CSV backward compatibility
            "semantic_score": ai_score.get("semantic_similarity_score", 0),
            "skill_score": ai_score.get("skill_match_score", 0),
            "role_score": ai_score.get("role_alignment_score", 0),
            "detected_role": ai_score.get("detected_role", "Unknown"),
            
            # Sub-similarities dynamically unpacked
            "skills_similarity": ai_score.get("skills_similarity", 0),
            "experience_similarity": ai_score.get("experience_similarity", 0),
            "projects_similarity": ai_score.get("projects_similarity", 0),
            
            # Context-aware evidence
            "strong_evidence": [
                {
                    "sentence": match["sentence"],
                    "relevance": f"{match['score']*100:.1f}%"
                }
                for match in evidence["strong_matches"]
            ],
            "weak_areas": [
                {
                    "sentence": area["sentence"],
                    "similarity": f"{area['score']*100:.1f}%"
                }
                for area in evidence["weak_areas"]
            ],
            "evidence_summary": evidence["evidence_summary"],
            "coverage_score": f"{evidence['coverage_score']*100:.1f}%",
            
            # Additional metadata
            "num_skills_matched": len(ai_score["exact_matches"]),
            "num_skills_required": len(required_skills),
            "skills_coverage": f"{(len(ai_score['exact_matches']) / len(required_skills) * 100) if required_skills else 0:.1f}%",
        }
        
        return result
    
    def rank_resumes(
        self,
        resumes_data: List[Dict[str, str]],
        job_role: str,
        requirements: str,
        job_description: str
    ) -> Dict[str, Any]:
        """
        Score and rank multiple resumes
        
        Args:
            resumes_data: List of dicts with 'name' and 'text' keys
            job_role: Target position
            requirements: Required skills
            job_description: Full job description
        
        Returns:
        {
            "total_resumes": int,
            "ranked_results": [
                {
                    "rank": 1,
                    "resume_name": "...",
                    "final_score": 85.0,
                    "matched_skills": [...],
                    "strong_evidence": [...],
                    ...
                },
                ...
            ],
            "summary": {
                "top_candidate": "...",
                "top_score": 85.0,
                "average_score": 65.0,
                "recommendation": "Top candidate recommended for interview"
            }
        }
        """
        
        if not resumes_data:
            return {
                "total_resumes": 0,
                "ranked_results": [],
                "summary": {
                    "top_candidate": None,
                    "top_score": 0,
                    "average_score": 0,
                    "recommendation": "No resumes to evaluate"
                }
            }
        
        # Score all resumes
        scored_resumes = []
        for resume in resumes_data:
            try:
                score = self.score_single_resume(
                    resume_text=resume["text"],
                    resume_name=resume["name"],
                    job_role=job_role,
                    requirements=requirements,
                    job_description=job_description
                )
                scored_resumes.append(score)
            except Exception as e:
                logger.error(f"Error scoring resume {resume['name']}: {e}")
                # Add failed resume with 0 score
                scored_resumes.append({
                    "resume_name": resume["name"],
                    "final_score": 0,
                    "recommendation": "❌ ERROR - Could not process resume",
                    "error": str(e)
                })
        
        # Sort by final_score descending
        ranked = sorted(
            scored_resumes,
            key=lambda x: x.get("final_score", 0),
            reverse=True
        )
        
        # Assign ranks
        for rank, resume in enumerate(ranked, 1):
            resume["rank"] = rank
        
        # Calculate summary statistics
        valid_scores = [r.get("final_score", 0) for r in ranked if "error" not in r]
        top_score = valid_scores[0] if valid_scores else 0
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        # Generate recommendation
        if top_score >= 75:
            summary_recommendation = "🌟 Top candidate strongly recommended for interview"
        elif top_score >= 60:
            summary_recommendation = "✅ Top candidate moderately recommended for interview"
        else:
            summary_recommendation = "⚠️ Limited suitable candidates found"
        
        return {
            "total_resumes": len(ranked),
            "ranked_results": ranked,
            "summary": {
                "top_candidate": ranked[0]["resume_name"] if ranked else None,
                "top_score": top_score,
                "average_score": round(avg_score, 1),
                "recommendation": summary_recommendation,
                "evaluation_date": self._get_timestamp()
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def export_ranking_csv(self, ranking_result: Dict[str, Any]) -> str:
        """
        Export ranking results to CSV format
        """
        import csv
        import io
        
        output = io.StringIO()
        
        if not ranking_result["ranked_results"]:
            return "No results to export"
        
        # CSV headers
        headers = [
            "Rank",
            "Resume Name",
            "Final Score",
            "Skills Match %",
            "Semantic Score",
            "Role Match",
            "Matched Skills Count",
            "Recommendation"
        ]
        
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        
        # Write rows
        for result in ranking_result["ranked_results"]:
            writer.writerow({
                "Rank": result["rank"],
                "Resume Name": result["resume_name"],
                "Final Score": f"{result.get('final_score', 0):.1f}",
                "Skills Match %": result.get("skill_score", 0),
                "Semantic Score": f"{result.get('semantic_score', 0):.1f}%",
                "Role Match": f"{result.get('role_score', 0):.1f}%",
                "Matched Skills Count": result.get("num_skills_matched", 0),
                "Recommendation": result.get("recommendation", "N/A")
            })
        
        return output.getvalue()
