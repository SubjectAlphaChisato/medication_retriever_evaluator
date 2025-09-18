#!/usr/bin/env python3
"""
Demonstration script for the enhanced hepatotoxicity extraction evaluation system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import HepatotoxicityPipeline


def main():
    """Run comprehensive evaluation demonstration."""
    
    print("🔬 HEPATOTOXICITY EXTRACTION EVALUATION DEMONSTRATION")
    print("🚀 OPTIMIZED: Single API call per drug for maximum efficiency")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = HepatotoxicityPipeline()
    
    # Get available drugs
    try:
        xml_files = pipeline.get_available_drugs()
        if not xml_files:
            print("❌ No XML files found in livertox/ directory")
            return
        
        print(f"📁 Found {len(xml_files)} drug files")
        
        # Demo 1: Evaluate specific question types
        print(f"\n{'DEMO 1: FOCUSED EVALUATION':^80}")
        print("-" * 80)
        print("Testing evaluation with specific question types...")
        
        test_files = xml_files[:2]  # First 2 drugs
        focused_report = pipeline.run_evaluation(
            test_files,
            "demo_focused_evaluation.json",
            question_types=["likelihood_assessment", "clinical_patterns", "dosing_safety"]
        )
        
        # Demo 2: Comprehensive evaluation (ALL fields)
        print(f"\n{'DEMO 2: COMPREHENSIVE EVALUATION (ALL FIELDS)':^80}")
        print("-" * 80)
        print("Testing comprehensive evaluation of ALL hepatotoxicity fields...")
        
        comprehensive_report = pipeline.run_evaluation(
            test_files,
            "demo_comprehensive_evaluation.json",
            question_types=None  # This will evaluate ALL fields
        )
        
        # Demo 3: Show actionable insights
        print(f"\n{'DEMO 3: ACTIONABLE INSIGHTS ANALYSIS':^80}")
        print("-" * 80)
        
        # Extract and display key insights
        if "detailed_results" in comprehensive_report:
            sample_result = comprehensive_report["detailed_results"][0]
            
            print(f"Sample Drug: {sample_result.get('drug_name', 'Unknown')}")
            print(f"Overall Score: {sample_result.get('overall_score', 0):.3f}")
            
            if sample_result.get('missing_information'):
                print("\n🔍 Missing Information Detected:")
                for missing in sample_result['missing_information']:
                    print(f"  • {missing}")
            
            if sample_result.get('hallucination_detected'):
                print("\n⚠️  Hallucinations Detected")
            
            feedback = sample_result.get('feedback', '')
            if feedback:
                print(f"\n📋 Evaluator Feedback:")
                print(f"  {feedback[:200]}..." if len(feedback) > 200 else f"  {feedback}")
        
        # Demo 4: Performance comparison
        print(f"\n{'DEMO 4: EVALUATION CONFIGURATION COMPARISON':^80}")
        print("-" * 80)
        
        print("Focused vs Comprehensive Evaluation Results:")
        print(f"  Focused Evaluation:")
        print(f"    - Questions: {focused_report['summary_statistics'].get('total_questions', 0)}")
        print(f"    - Avg Score: {focused_report['summary_statistics'].get('average_overall_score', 0):.3f}")
        
        print(f"  Comprehensive Evaluation:")
        print(f"    - Questions: {comprehensive_report['summary_statistics'].get('total_questions', 0)}")
        print(f"    - Avg Score: {comprehensive_report['summary_statistics'].get('average_overall_score', 0):.3f}")
        
        # Show available question types
        print(f"\n{'AVAILABLE QUESTION TYPES':^80}")
        print("-" * 80)
        available_types = [
            "core_identification",
            "likelihood_assessment", 
            "clinical_patterns",
            "dosing_safety",
            "laboratory_values",
            "risk_assessment",
            "regulatory_status"
        ]
        
        for i, qtype in enumerate(available_types, 1):
            print(f"  {i}. {qtype}")
        
        print(f"\n{'USAGE EXAMPLES':^80}")
        print("-" * 80)
        print("1. Evaluate ALL fields (default):")
        print("   python -m src.pipeline evaluate --limit 3")
        print()
        print("2. Evaluate specific aspects:")
        print("   python -m src.pipeline evaluate --limit 3 --question-types likelihood_assessment clinical_patterns")
        print()
        print("3. Extract and evaluate single drug:")
        print("   python -m src.pipeline extract --drug Acetaminophen")
        print("   python -m src.pipeline evaluate --limit 1 --question-types dosing_safety")
        
        print(f"\n{'SUMMARY':^80}")
        print("=" * 80)
        print("✅ Enhanced evaluation system successfully demonstrated!")
        print("📊 Evaluation reports saved:")
        print("   - demo_focused_evaluation.json")
        print("   - demo_comprehensive_evaluation.json")
        print("🔧 The evaluator now provides:")
        print("   - Configurable question types")
        print("   - Comprehensive field evaluation")
        print("   - Actionable insights and recommendations")
        print("   - Detailed performance analytics")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
