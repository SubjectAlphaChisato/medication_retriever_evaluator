"""
Evaluator agent for testing the accuracy of the hepatotoxicity retriever.
"""

import json
import re
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from datetime import datetime

from openai import OpenAI
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from .models import HepatotoxicityData, EvaluationQuestion, EvaluationResult
from .retriever import HepatotoxicityRetriever
from .xml_parser import LiverToxXMLParser
from .config import DEFAULT_EVALUATOR_MODEL, EVALUATION_QUESTIONS_PER_DRUG, OPENAI_API_KEY, ANTHROPIC_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HepatotoxicityEvaluator:
    """Agent for evaluating the accuracy of hepatotoxicity data extraction."""
    
    def __init__(self, model: str = DEFAULT_EVALUATOR_MODEL, retriever_model: Optional[str] = None):
        """Initialize the evaluator with LLM client."""
        self.model = model or DEFAULT_EVALUATOR_MODEL
        logger.info(f"Initializing evaluator with model: {self.model}")
        
        self.retriever = HepatotoxicityRetriever(retriever_model or self.model)
        
        # Initialize appropriate client based on model
        if self.model and self.model.startswith("claude"):
            if not ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not found")
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
            self.provider = "anthropic"
        else:
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.provider = "openai"
            # Default to GPT-4o if model is None or not recognized
            if not self.model or not self.model.startswith("gpt"):
                self.model = "gpt-4o"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_llm(self, messages: list) -> str:
        """Call LLM API with retry logic."""
        try:
            if self.provider == "anthropic":
                # Convert OpenAI-style messages to Anthropic format
                system_message = ""
                user_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    elif msg["role"] == "user":
                        user_messages.append(msg["content"])
                
                combined_user_message = "\n\n".join(user_messages)
                
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.1,
                    system=system_message,
                    messages=[{"role": "user", "content": combined_user_message}]
                )
                return response.content[0].text
            else:
                # OpenAI
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4000
                )
                return response.choices[0].message.content
        except Exception as e:
            print(traceback.format_exc())
            logger.error(f"Error calling {self.provider} API ({self.model}): {e}")
            raise
    
    def generate_evaluation_questions(
        self, 
        drug_name: str,
        question_types: Optional[List[str]] = None
    ) -> List[EvaluationQuestion]:
        """Generate evaluation questions for a specific drug."""
        
        # Default: evaluate ALL fields
        if question_types is None:
            question_types = [
                "core_identification",
                "likelihood_assessment", 
                "clinical_patterns",
                "dosing_safety",
                "laboratory_values",
                "risk_assessment",
                "regulatory_status"
            ]
        
        all_questions = {
            "core_identification": EvaluationQuestion(
                question=f"Extract the core identification information for {drug_name}",
                expected_fields=["drug_name"],
                drug_name=drug_name
            ),
            "likelihood_assessment": EvaluationQuestion(
                question=f"What is the DILI likelihood score and evidence assessment for {drug_name}?",
                expected_fields=["dili_likelihood_score"],
                drug_name=drug_name
            ),
            "clinical_patterns": EvaluationQuestion(
                question=f"What are the clinical patterns and injury characteristics for {drug_name}?",
                expected_fields=["injury_pattern", "is_immune_mediated"],
                drug_name=drug_name
            ),
            "dosing_safety": EvaluationQuestion(
                question=f"What are the safe and toxic dosing parameters for {drug_name}?",
                expected_fields=["safe_dose", "toxic_dose", "onset_time"],
                drug_name=drug_name
            ),
            "laboratory_values": EvaluationQuestion(
                question=f"What are the laboratory findings and enzyme elevations for {drug_name}?",
                expected_fields=["peak_alt", "peak_alp", "r_ratio", "bilirubin_peak", 
                               "fraction_patients_with_enzyme_elevation", "fraction_patients_with_dili"],
                drug_name=drug_name
            ),
            "risk_assessment": EvaluationQuestion(
                question=f"What are the risk factors and patient considerations for {drug_name}?",
                expected_fields=["risk_factors"],
                drug_name=drug_name
            ),
            "regulatory_status": EvaluationQuestion(
                question=f"What is the regulatory and approval status for {drug_name}?",
                expected_fields=["regulatory_status"],
                drug_name=drug_name
            )
        }
        
        return [all_questions[qt] for qt in question_types if qt in all_questions]
    
    def evaluate_retriever_response(
        self, 
        question: EvaluationQuestion, 
        retriever_response: HepatotoxicityData,
        xml_file_path: str
    ) -> EvaluationResult:
        """Evaluate a retriever response against the source XML and external verification."""
        
        # Parse the original XML for ground truth comparison
        parser = LiverToxXMLParser(xml_file_path)
        full_text = parser.extract_text_content()
        sections = parser.extract_sections()
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(
            question, retriever_response, full_text, sections
        )
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert medical data quality evaluator specializing in hepatotoxicity information extraction. Your role is to provide comprehensive, actionable evaluation of extraction systems.

**EVALUATION METHODOLOGY:**

1. **Source Verification**: Compare every extracted data point against the source XML
2. **Completeness Analysis**: Identify information present in source but missed by extractor
3. **Hallucination Detection**: Flag any information not supported by the source
4. **Quality Assessment**: Evaluate the reasoning quality, especially for inferred fields
5. **Actionable Insights**: Provide specific, implementable improvement recommendations

**EVALUATION CRITERIA:**

- **Accuracy** (0.0-1.0): How well does extracted data match the source?
- **Completeness** (0.0-1.0): How much available information was successfully extracted?
- **Precision** (0.0-1.0): How well does the extractor avoid hallucinations?
- **Reasoning Quality** (0.0-1.0): For inferred fields (like DILI scores), how sound is the logic?

**ACTIONABLE FEEDBACK REQUIREMENTS:**

Your feedback must include:
1. **Specific Issues**: Exact problems with field-level detail
2. **Root Cause Analysis**: Why extraction failed (prompt issue, model limitation, etc.)
3. **Improvement Strategies**: Concrete steps to enhance performance
4. **Pattern Recognition**: Identify systematic vs. isolated issues

Return detailed, structured evaluation as JSON with actionable insights."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response_text = self._call_llm(messages)
        
        # Parse evaluation response
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_text = response_text
            
            eval_data = json.loads(json_text)
            
            # Create evaluation result
            result = EvaluationResult(
                question=question.question,
                drug_name=question.drug_name,
                retriever_response=retriever_response,
                correctness_scores=eval_data.get("field_scores", {}),
                overall_score=eval_data.get("overall_score", 0.0),
                feedback=eval_data.get("feedback", ""),
                hallucination_detected=eval_data.get("hallucination_detected", False),
                missing_information=eval_data.get("missing_information", [])
            )
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation JSON: {e}")
            logger.error(f"Response text: {response_text}")
            
            # Create a fallback result
            return EvaluationResult(
                question=question.question,
                drug_name=question.drug_name,
                retriever_response=retriever_response,
                correctness_scores={},
                overall_score=0.0,
                feedback=f"Evaluation failed: {e}",
                hallucination_detected=False,
                missing_information=[]
            )
    
    def evaluate_retriever_comprehensive(
        self,
        questions: List[EvaluationQuestion],
        retriever_response: HepatotoxicityData,
        xml_file_path: str
    ) -> str:
        """Evaluate all questions in a single comprehensive API call and return plain text report."""
        
        # Parse the original XML for ground truth comparison
        parser = LiverToxXMLParser(xml_file_path)
        full_text = parser.extract_text_content()
        sections = parser.extract_sections()
        
        # Create comprehensive evaluation prompt
        prompt = self._create_comprehensive_evaluation_prompt(
            questions, retriever_response, full_text, sections
        )
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert medical data quality evaluator specializing in hepatotoxicity information extraction. 

Generate a comprehensive, plain-language evaluation report that assesses how well the extraction system performed. Your report should be clear, actionable, and easy to understand.

**REPORT STRUCTURE:**
1. **Executive Summary** - Overall performance assessment
2. **Field-by-Field Analysis** - Detailed review of each extracted field
3. **Accuracy Assessment** - What was correct vs incorrect
4. **Completeness Review** - What information was missed
5. **Hallucination Detection** - Any fabricated information
6. **Actionable Recommendations** - Specific improvements needed
7. **Overall Score** - Rate performance from 1-10

Write in clear, professional language. Be specific about issues and provide concrete suggestions for improvement."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response_text = self._call_llm(messages)
        return response_text
    
    def _split_comprehensive_result(
        self, 
        comprehensive_report: str, 
        questions: List[EvaluationQuestion]
    ) -> List[EvaluationResult]:
        """Convert comprehensive text report into individual question results."""
        
        results = []
        from .models import HepatotoxicityData
        
        # Extract overall score from the report (look for "Score: X/10" or similar)
        score_match = re.search(r'(?:overall score|score):\s*(\d+(?:\.\d+)?)\s*/?\s*10', comprehensive_report, re.IGNORECASE)
        overall_score = float(score_match.group(1)) / 10.0 if score_match else 0.8  # Default to 0.8 if not found
        
        # Create a result for each question with the same overall assessment
        for question in questions:
            dummy_data = HepatotoxicityData(drug_name=question.drug_name)
            
            # Create basic scores for each field
            question_scores = {field: overall_score for field in question.expected_fields}
            
            result = EvaluationResult(
                question=question.question,
                drug_name=question.drug_name,
                retriever_response=dummy_data,
                correctness_scores=question_scores,
                overall_score=overall_score,
                feedback=comprehensive_report,  # Use the full report as feedback
                hallucination_detected="hallucination" in comprehensive_report.lower(),
                missing_information=[]  # Could extract this from report if needed
            )
            
            results.append(result)
        
        return results
    
    def _create_comprehensive_evaluation_prompt(
        self,
        questions: List[EvaluationQuestion],
        retriever_response: HepatotoxicityData,
        full_text: str,
        sections: Dict[str, str]
    ) -> str:
        """Create comprehensive evaluation prompt for all questions at once."""
        
        # Limit text size
        max_text_length = 12000
        if len(full_text) > max_text_length:
            full_text = full_text[:max_text_length] + "... [truncated]"
        
        # Get all extracted data
        all_extracted_data = retriever_response.model_dump()
        
        # Format questions
        questions_text = "\n".join([
            f"{i+1}. {q.question} (Focus: {', '.join(q.expected_fields)})"
            for i, q in enumerate(questions)
        ])
        
        return f"""
**COMPREHENSIVE EVALUATION TASK:**

Please evaluate how well the extraction system performed on the drug: **{questions[0].drug_name if questions else 'Unknown'}**

**QUESTIONS TO EVALUATE:**
{questions_text}

**EXTRACTED DATA:**
{json.dumps(all_extracted_data, indent=2, default=str)}

**SOURCE DOCUMENT (for verification):**
{full_text[:8000]}...

**EVALUATION INSTRUCTIONS:**

Please write a comprehensive evaluation report that covers:

1. **Executive Summary**: Overall assessment of extraction quality
2. **Field-by-Field Analysis**: Review each extracted field for accuracy
3. **Accuracy Assessment**: Compare extracted data against source document
4. **Completeness Review**: What important information was missed?
5. **Hallucination Check**: Any information not supported by the source?
6. **Actionable Recommendations**: Specific ways to improve extraction
7. **Overall Score**: Rate the extraction quality from 1-10

**KEY FIELDS TO EVALUATE:**
- drug_name
- dili_likelihood_score  
- injury_pattern
- fraction_patients_with_enzyme_elevation
- fraction_patients_with_dili
- risk_factors
- safe_dose, toxic_dose
- onset_time
- peak_alt, peak_alp, r_ratio, bilirubin_peak
- is_immune_mediated
- regulatory_status

**EVALUATION CRITERIA:**
- Is the information accurate based on the source?
- Is the extraction complete or are key details missing?
- Are inferences (like DILI scores) well-reasoned?
- Is any information fabricated or unsupported?

Write your evaluation as a clear, detailed report. End with an overall score out of 10."""
    
    def _create_evaluation_prompt(
        self, 
        question: EvaluationQuestion, 
        retriever_response: HepatotoxicityData,
        full_text: str,
        sections: Dict[str, str]
    ) -> str:
        """Create evaluation prompt for the LLM."""
        
        # Limit text size
        max_text_length = 12000
        if len(full_text) > max_text_length:
            full_text = full_text[:max_text_length] + "... [truncated]"
        
        # Evaluate ALL fields, not just the ones in the question
        all_extracted_data = retriever_response.model_dump()
        
        # Get relevant fields for this question (for focused analysis)
        relevant_fields = question.expected_fields
        focused_data = {}
        for field in relevant_fields:
            value = getattr(retriever_response, field, None)
            focused_data[field] = value
        
        return f"""
**COMPREHENSIVE EVALUATION TASK:**
Primary Question: "{question.question}"
Drug: {question.drug_name}
Focus Fields: {relevant_fields}

**ALL EXTRACTED DATA (evaluate comprehensively):**
```json
{json.dumps(all_extracted_data, indent=2, default=str)}
```

**PRIMARY FOCUS DATA:**
```json
{json.dumps(focused_data, indent=2, default=str)}
```

**SOURCE DOCUMENT SECTIONS:**
{json.dumps(sections, indent=2)}

**FULL SOURCE TEXT:**
{full_text}

**COMPREHENSIVE EVALUATION REQUIREMENTS:**

Evaluate **ALL 14 FIELDS** from the extracted data:
1. drug_name
2. dili_likelihood_score  
3. injury_pattern
4. fraction_patients_with_enzyme_elevation
5. fraction_patients_with_dili
6. risk_factors
7. safe_dose
8. toxic_dose
9. onset_time
10. peak_alt
11. peak_alp
12. r_ratio
13. bilirubin_peak
14. is_immune_mediated
15. regulatory_status

**EVALUATION DIMENSIONS:**

For EACH field, assess:
- **Accuracy**: Does extracted value match source evidence?
- **Source Support**: Is there clear textual support in the document?
- **Appropriateness**: Is null used correctly when information is unavailable?
- **Reasoning Quality**: For inferred fields, is the logic sound?

**SPECIAL ATTENTION:**

1. **DILI Likelihood Score**: Evaluate reasoning process, evidence synthesis
2. **Quantitative Fields**: Check numerical accuracy and unit consistency  
3. **Risk Factors**: Verify quotes and supporting evidence
4. **Dosing Information**: Validate dose values and units
5. **Laboratory Values**: Check for proper numeric extraction

**ACTIONABLE INSIGHTS REQUIRED:**

For each problematic field:
1. **Specific Issue**: What exactly went wrong?
2. **Root Cause**: Why did it fail? (prompt clarity, model limitation, parsing issue)
3. **Improvement Strategy**: Concrete fix (better prompt, different approach, etc.)
4. **Priority Level**: High/Medium/Low for fixing

**OUTPUT FORMAT:**
Return comprehensive JSON evaluation:

```json
{{
    "field_evaluations": {{
        "drug_name": {{
            "score": 0.0_to_1.0,
            "accuracy": 0.0_to_1.0,
            "completeness": 0.0_to_1.0,
            "source_support": true_or_false,
            "issues": "specific problems found",
            "root_cause": "why it failed",
            "improvement_strategy": "how to fix it",
            "priority": "high|medium|low"
        }},
        "dili_likelihood_score": {{ ... }},
        "injury_pattern": {{ ... }},
        ... [continue for all 15 fields]
    }},
    "overall_scores": {{
        "accuracy": average_accuracy_across_fields,
        "completeness": average_completeness_across_fields,
        "precision": hallucination_avoidance_score,
        "reasoning_quality": inference_quality_score
    }},
    "critical_issues": ["list", "of", "high", "priority", "problems"],
    "systematic_patterns": ["recurring", "issues", "across", "fields"],
    "actionable_recommendations": [
        "Specific improvement 1",
        "Specific improvement 2",
        "..."
    ],
    "missing_information": ["information", "present", "in", "source", "but", "not", "extracted"],
    "hallucinations": ["extracted", "information", "not", "in", "source"],
    "strengths": ["what", "the", "extractor", "did", "well"],
    "overall_assessment": "comprehensive summary with improvement roadmap"
}}
```

**CRITICAL**: Evaluate ALL fields comprehensively, not just the focus fields. Provide specific, actionable feedback for building a better extraction system."""
    
    def run_comprehensive_evaluation(
        self, 
        xml_files: List[str], 
        output_file: Optional[str] = None,
        question_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation on multiple drugs."""
        
        all_results = []
        summary_stats = {
            "total_drugs": len(xml_files),
            "total_questions": 0,
            "average_overall_score": 0.0,
            "field_performance": {},
            "hallucination_rate": 0.0,
            "drugs_evaluated": []
        }
        
        for xml_file in xml_files:
            drug_name = Path(xml_file).stem
            logger.info(f"Evaluating {drug_name}...")
            
            try:
                # Extract data with retriever
                extraction_result = self.retriever.extract_hepatotoxicity_data(xml_file)
                
                # Generate questions
                questions = self.generate_evaluation_questions(drug_name, question_types)
                
                # Single comprehensive evaluation for all questions
                eval_report = self.evaluate_retriever_comprehensive(
                    questions, extraction_result.data, xml_file
                )
                
                # Save the full text report
                drug_report_file = f"{drug_name}_evaluation_report.txt"
                with open(drug_report_file, 'w', encoding='utf-8') as f:
                    f.write(f"EVALUATION REPORT: {drug_name}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(eval_report)
                logger.info(f"Detailed report saved to {drug_report_file}")
                
                # Convert text report to individual question results for compatibility
                drug_results = self._split_comprehensive_result(eval_report, questions)
                all_results.extend(drug_results)
                
                # Track drug performance
                drug_score = sum(r.overall_score for r in drug_results) / len(drug_results)
                summary_stats["drugs_evaluated"].append({
                    "drug_name": drug_name,
                    "average_score": drug_score,
                    "questions_count": len(drug_results)
                })
                
                logger.info(f"Completed {drug_name}: Average score {drug_score:.2f}")
                
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                logger.error(f"Failed to evaluate {drug_name}: {e}")
                continue
        
        # Calculate summary statistics
        if all_results:
            summary_stats["total_questions"] = len(all_results)
            summary_stats["average_overall_score"] = sum(r.overall_score for r in all_results) / len(all_results)
            summary_stats["hallucination_rate"] = sum(1 for r in all_results if r.hallucination_detected) / len(all_results)
            
            # Field performance
            field_scores = {}
            for result in all_results:
                for field, score in result.correctness_scores.items():
                    if field not in field_scores:
                        field_scores[field] = []
                    field_scores[field].append(score)
            
            for field, scores in field_scores.items():
                summary_stats["field_performance"][field] = {
                    "average_score": sum(scores) / len(scores),
                    "count": len(scores)
                }
        
        # Create comprehensive report
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "summary_statistics": summary_stats,
            "detailed_results": [r.model_dump() for r in all_results],
            "recommendations": self._generate_recommendations(summary_stats, all_results)
        }
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Evaluation report saved to {output_file}")
        
        return report
    
    def _generate_recommendations(
        self, 
        summary_stats: Dict[str, Any], 
        results: List[EvaluationResult]
    ) -> List[str]:
        """Generate recommendations based on evaluation results."""
        
        recommendations = []
        
        # Overall performance
        avg_score = summary_stats.get("average_overall_score", 0.0)
        if avg_score < 0.7:
            recommendations.append(
                f"Overall performance is below acceptable threshold ({avg_score:.2f} < 0.7). "
                "Consider improving prompt engineering or using a more capable model."
            )
        
        # Hallucination rate
        hallucination_rate = summary_stats.get("hallucination_rate", 0.0)
        if hallucination_rate > 0.1:
            recommendations.append(
                f"High hallucination rate detected ({hallucination_rate:.1%}). "
                "Improve prompts to emphasize conservative extraction and source verification."
            )
        
        # Field-specific performance
        field_perf = summary_stats.get("field_performance", {})
        poor_fields = [field for field, stats in field_perf.items() 
                      if stats["average_score"] < 0.6]
        
        if poor_fields:
            recommendations.append(
                f"Poor performance on fields: {', '.join(poor_fields)}. "
                "Consider field-specific prompt improvements or additional training data."
            )
        
        # Missing information patterns
        missing_patterns = {}
        for result in results:
            for missing in result.missing_information:
                missing_patterns[missing] = missing_patterns.get(missing, 0) + 1
        
        if missing_patterns:
            top_missing = sorted(missing_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            recommendations.append(
                f"Frequently missed information: {', '.join([item[0] for item in top_missing])}. "
                "Review extraction logic for these patterns."
            )
        
        return recommendations
