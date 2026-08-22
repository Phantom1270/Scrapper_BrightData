"""
Evaluation Markdown Report Generator.
"""

from typing import List
from pathlib import Path
from datetime import datetime

from rag.evaluation.evaluator import EvaluationReport


class ReportGenerator:
    """Generates a human-readable markdown report from evaluation results."""

    def __init__(self, settings=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()
            
        self.report_path = settings.evaluation.report_path

    def generate_report(self, report: EvaluationReport) -> str:
        """Generate a markdown report string."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_set_name = report.metadata.get("name", "Unknown Test Set")
        
        # Get model name from first result if available
        model_name = "Unknown"
        if report.results:
            model_name = report.results[0].llm_model
            
        md = []
        md.append("# RAG Pipeline Evaluation Report\n")
        md.append(f"**Generated:** {timestamp}")
        md.append(f"**Test Set:** {test_set_name}")
        md.append(f"**Total Questions:** {report.total_questions}")
        md.append(f"**LLM Model:** {model_name}\n")
        md.append("---\n")
        
        # Overall Metrics
        md.append("## Overall Metrics\n")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        
        agg = report.aggregate_metrics
        
        def fmt_pct(val):
            return f"{val*100:.1f}%" if isinstance(val, (float, int)) else "N/A"
            
        def fmt_dec(val):
            return f"{val:.3f}" if isinstance(val, (float, int)) else "N/A"
            
        md.append(f"| Precision@1 | {fmt_dec(agg.get('retrieval_precision@1'))} |")
        md.append(f"| Precision@3 | {fmt_dec(agg.get('retrieval_precision@3'))} |")
        md.append(f"| Precision@5 | {fmt_dec(agg.get('retrieval_precision@5'))} |")
        md.append(f"| Recall@3 | {fmt_dec(agg.get('retrieval_recall@3'))} |")
        md.append(f"| Recall@5 | {fmt_dec(agg.get('retrieval_recall@5'))} |")
        md.append(f"| MRR | {fmt_dec(agg.get('retrieval_mrr'))} |")
        md.append(f"| nDCG@5 | {fmt_dec(agg.get('retrieval_ndcg@5'))} |")
        md.append(f"| Keyword Coverage | {fmt_dec(agg.get('generation_keyword_coverage'))} |")
        md.append(f"| Faithfulness | {fmt_dec(agg.get('generation_faithfulness'))} |")
        md.append(f"| Answer Relevance | {fmt_dec(agg.get('generation_answer_relevance'))} |")
        md.append(f"| Citation Rate | {fmt_pct(agg.get('generation_has_citations_rate'))} |\n")
        md.append("---\n")
        
        # Results by Category
        md.append("## Results by Category\n")
        for cat, cat_agg in sorted(report.category_metrics.items()):
            count = cat_agg.get('count', 0)
            md.append(f"### {cat.title()} Questions ({count} questions)")
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            md.append(f"| Precision@5 | {fmt_dec(cat_agg.get('retrieval_precision@5'))} |")
            md.append(f"| Recall@5 | {fmt_dec(cat_agg.get('retrieval_recall@5'))} |")
            md.append(f"| Keyword Coverage | {fmt_dec(cat_agg.get('generation_keyword_coverage'))} |\n")
            
        md.append("---\n")
        
        # Results by Difficulty
        md.append("## Results by Difficulty\n")
        for diff, diff_agg in sorted(report.difficulty_metrics.items()):
            count = diff_agg.get('count', 0)
            md.append(f"### {diff.title()} ({count} questions)")
            md.append(f"- Precision@5: {fmt_dec(diff_agg.get('retrieval_precision@5'))}")
            md.append(f"- Keyword Coverage: {fmt_dec(diff_agg.get('generation_keyword_coverage'))}\n")
            
        md.append("---\n")
        
        # Individual Results
        md.append("## Individual Results\n")
        for i, res in enumerate(report.results):
            md.append(f"### Q{i+1}: {res.question}")
            md.append(f"- **Category:** {res.category} | **Difficulty:** {res.difficulty}")
            
            ret_count = len(res.retrieved_chunk_ids)
            exp_count = len(res.expected_chunk_ids)
            overlap = res.retrieval_metrics.get('overlap_count', 0)
            
            md.append(f"- **Retrieved:** {ret_count} chunks | **Expected:** {exp_count} chunks | **Overlap:** {overlap}")
            
            p5 = fmt_dec(res.retrieval_metrics.get('precision@5'))
            r5 = fmt_dec(res.retrieval_metrics.get('recall@5'))
            mrr = fmt_dec(res.retrieval_metrics.get('mrr'))
            
            md.append(f"- **Precision@5:** {p5} | **Recall@5:** {r5} | **MRR:** {mrr}")
            
            ans_preview = res.generated_answer[:200].replace('\n', ' ')
            if len(res.generated_answer) > 200:
                ans_preview += "..."
                
            md.append(f"- **Answer:** {ans_preview}")
            
            kw = fmt_dec(res.generation_metrics.get('keyword_coverage'))
            fth = fmt_dec(res.generation_metrics.get('faithfulness'))
            
            md.append(f"- **Keyword Coverage:** {kw} | **Faithfulness:** {fth}")
            md.append(f"- **Confidence:** {res.confidence}")
            md.append(f"- **Time:** {res.retrieval_time_ms:.1f}ms retrieval + {res.generation_time_ms:.1f}ms generation\n")
            
        md.append("---\n")
        
        # Observations
        md.append("## Observations\n")
        observations = self._generate_observations(report)
        for obs in observations:
            md.append(f"- {obs}")
            
        return "\n".join(md)

    def save_report(self, report_text: str, path: str = None) -> str:
        """Save the report to a markdown file."""
        if path is None:
            path = self.report_path
            
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        return str(path)

    def _generate_observations(self, report: EvaluationReport) -> List[str]:
        """Generate automated observations from the data."""
        observations = []
        
        if not report.results:
            return ["No results to observe."]
            
        # Best/worst categories by precision@5
        cats = report.category_metrics
        if cats:
            valid_cats = {k: v.get('retrieval_precision@5', 0) for k, v in cats.items() if v.get('retrieval_precision@5') is not None}
            if valid_cats:
                best_cat = max(valid_cats.items(), key=lambda x: x[1])
                worst_cat = min(valid_cats.items(), key=lambda x: x[1])
                
                observations.append(f"Best performing category (Precision@5): **{best_cat[0]}** ({best_cat[1]:.3f})")
                observations.append(f"Worst performing category (Precision@5): **{worst_cat[0]}** ({worst_cat[1]:.3f})")
                
                # Warnings
                for cat, p5 in valid_cats.items():
                    if p5 < 0.3:
                        observations.append(f"⚠️ Warning: Category **{cat}** has poor retrieval performance (Precision@5 < 0.3).")
                        
        # Zero retrieval overlap
        zero_overlap = sum(1 for r in report.results if r.retrieval_metrics.get('overlap_count', 0) == 0)
        observations.append(f"Questions with zero retrieval overlap: **{zero_overlap}** out of {report.total_questions}")
        
        # Refused answers
        refused = sum(1 for r in report.results if r.generation_metrics.get('refused_answer', False))
        observations.append(f"Questions where the LLM refused to answer: **{refused}** out of {report.total_questions}")
        
        # Timing
        avg_gen_time = report.aggregate_metrics.get('generation_generation_time_ms', 0)
        observations.append(f"Average generation time: **{avg_gen_time:.1f}ms**")
        
        return observations
