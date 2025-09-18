"""
Main pipeline for hepatotoxicity data extraction and evaluation.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
import logging

from .retriever import HepatotoxicityRetriever
from .evaluator import HepatotoxicityEvaluator
from .config import LIVERTOX_DATA_DIR, DEFAULT_EVALUATOR_MODEL, DEFAULT_RETRIEVER_MODEL

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HepatotoxicityPipeline:
    """Main pipeline for hepatotoxicity data extraction and evaluation."""
    
    def __init__(self, retriever_model: Optional[str] = DEFAULT_RETRIEVER_MODEL, evaluator_model: Optional[str] = DEFAULT_EVALUATOR_MODEL):
        """Initialize the pipeline with configurable models."""
        self.retriever = HepatotoxicityRetriever(retriever_model)
        self.evaluator = HepatotoxicityEvaluator(evaluator_model, retriever_model)
    
    def extract_single_drug(self, xml_file: str, output_file: Optional[str] = None) -> dict:
        """Extract hepatotoxicity data for a single drug."""
        logger.info(f"Extracting data from {xml_file}")
        
        try:
            result = self.retriever.extract_hepatotoxicity_data(xml_file)
            
            # Convert to dict for JSON serialization
            result_dict = {
                "extraction_result": result.model_dump(),
                "file_path": xml_file
            }
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                logger.info(f"Results saved to {output_file}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Failed to extract data from {xml_file}: {e}")
            raise
    
    def extract_multiple_drugs(
        self, 
        xml_files: List[str], 
        output_dir: str = "output"
    ) -> List[dict]:
        """Extract hepatotoxicity data for multiple drugs."""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        
        for xml_file in xml_files:
            drug_name = Path(xml_file).stem
            output_file = output_path / f"{drug_name}_extraction.json"
            
            try:
                result = self.extract_single_drug(xml_file, str(output_file))
                results.append(result)
                logger.info(f"Successfully processed {drug_name}")
                
            except Exception as e:
                logger.error(f"Failed to process {drug_name}: {e}")
                continue
        
        # Save summary
        summary_file = output_path / "extraction_summary.json"
        summary = {
            "total_files": len(xml_files),
            "successful_extractions": len(results),
            "failed_extractions": len(xml_files) - len(results),
            "results": results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Extraction summary saved to {summary_file}")
        return results
    
    def run_evaluation(
        self, 
        xml_files: List[str], 
        output_file: str = "evaluation_report.json",
        question_types: Optional[List[str]] = None
    ) -> dict:
        """Run comprehensive evaluation on the retriever."""
        logger.info(f"Starting evaluation on {len(xml_files)} drugs")
        logger.info(f"Question types: {question_types or 'ALL (default)'}")
        logger.info("Using optimized single API call per drug for efficiency")
        
        report = self.evaluator.run_comprehensive_evaluation(
            xml_files, output_file, question_types
        )
        
        # Print detailed summary
        stats = report["summary_statistics"]
        print("\n" + "="*80)
        print("COMPREHENSIVE HEPATOTOXICITY EXTRACTION EVALUATION")
        print("="*80)
        print(f"Evaluation Date: {report['evaluation_timestamp']}")
        print(f"Total Drugs Evaluated: {stats['total_drugs']}")
        print(f"Total Questions Asked: {stats['total_questions']}")
        print(f"Average Overall Score: {stats['average_overall_score']:.3f}")
        print(f"Hallucination Rate: {stats['hallucination_rate']:.1%}")
        
        print(f"\n{'FIELD PERFORMANCE ANALYSIS':^80}")
        print("-" * 80)
        field_perf = stats.get("field_performance", {})
        
        # Sort fields by performance for better readability
        sorted_fields = sorted(field_perf.items(), key=lambda x: x[1]['average_score'], reverse=True)
        
        print(f"{'Field':<35} {'Score':<10} {'Samples':<10} {'Status'}")
        print("-" * 80)
        for field, perf in sorted_fields:
            score = perf['average_score']
            count = perf['count']
            status = "🟢 Good" if score >= 0.8 else "🟡 Fair" if score >= 0.6 else "🔴 Poor"
            print(f"{field:<35} {score:<10.3f} {count:<10} {status}")
        
        print(f"\n{'DRUG-BY-DRUG PERFORMANCE':^80}")
        print("-" * 80)
        for drug_data in stats.get("drugs_evaluated", []):
            score = drug_data['average_score']
            status = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
            print(f"{drug_data['drug_name']:<30} {score:.3f} {status}")
        
        print(f"\n{'ACTIONABLE RECOMMENDATIONS':^80}")
        print("-" * 80)
        for i, rec in enumerate(report.get("recommendations", []), 1):
            print(f"{i:2d}. {rec}")
        
        print("\n" + "="*80)
        print(f"📊 Detailed evaluation report saved to: {output_file}")
        print("="*80)
        
        return report
    
    def get_available_drugs(self, data_dir: str = LIVERTOX_DATA_DIR) -> List[str]:
        """Get list of available XML files."""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        xml_files = list(data_path.glob("*.xml"))
        return [str(f) for f in xml_files]


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(description="Hepatotoxicity Data Extraction Pipeline")
    parser.add_argument("command", choices=["extract", "evaluate", "demo"], 
                       help="Command to run")
    parser.add_argument("--drug", type=str, help="Specific drug to process (for extract)")
    parser.add_argument("--limit", type=int, default=5, 
                       help="Limit number of drugs to process")
    parser.add_argument("--output", type=str, help="Output file or directory")
    parser.add_argument("--data-dir", type=str, default=LIVERTOX_DATA_DIR,
                       help="Directory containing XML files")
    parser.add_argument("--question-types", type=str, nargs="*",
                       help="Question types for evaluation (default: all)")
    parser.add_argument("--retriever-model", type=str, 
                       help="Model for retriever (default: gpt-4o)")
    parser.add_argument("--evaluator-model", type=str,
                       help="Model for evaluator (default: gpt-4o)")
    parser.add_argument("--model", type=str,
                       help="Model for both retriever and evaluator")
    
    args = parser.parse_args()
    
    # Determine models to use
    retriever_model = args.retriever_model or args.model or DEFAULT_RETRIEVER_MODEL
    evaluator_model = args.evaluator_model or args.model or DEFAULT_EVALUATOR_MODEL
    
    # Log model configuration
    if retriever_model or evaluator_model:
        logger.info(f"Using custom models - Retriever: {retriever_model or 'default'}, Evaluator: {evaluator_model or 'default'}")
    
    pipeline = HepatotoxicityPipeline(retriever_model, evaluator_model)
    
    try:
        if args.command == "extract":
            if args.drug:
                # Extract single drug
                xml_file = Path(args.data_dir) / f"{args.drug}.xml"
                if not xml_file.exists():
                    print(f"File not found: {xml_file}")
                    return
                
                output_file = args.output or f"{args.drug}_extraction.json"
                result = pipeline.extract_single_drug(str(xml_file), output_file)
                print(f"Extraction completed. Results saved to {output_file}")
                
            else:
                # Extract multiple drugs
                xml_files = pipeline.get_available_drugs(args.data_dir)[:args.limit]
                output_dir = args.output or "output"
                results = pipeline.extract_multiple_drugs(xml_files, output_dir)
                print(f"Extracted data for {len(results)} drugs. Results in {output_dir}/")
        
        elif args.command == "evaluate":
            if args.drug:
                # Evaluate specific drug
                xml_file = Path(args.data_dir) / f"{args.drug}.xml"
                if not xml_file.exists():
                    print(f"File not found: {xml_file}")
                    return
                
                xml_files = [str(xml_file)]
                output_file = args.output or f"{args.drug}_evaluation.json"
            else:
                # Evaluate multiple drugs
                xml_files = pipeline.get_available_drugs(args.data_dir)[:args.limit]
                output_file = args.output or "evaluation_report.json"
            
            pipeline.run_evaluation(xml_files, output_file, args.question_types)
        
        elif args.command == "demo":
            # Demo mode - extract and evaluate a few drugs
            print("Running demonstration...")
            
            # Get a few drugs for demo
            xml_files = pipeline.get_available_drugs(args.data_dir)[:3]
            print(f"Processing {len(xml_files)} drugs for demo...")
            
            # Extract data
            print("\n1. Extracting data...")
            results = pipeline.extract_multiple_drugs(xml_files, "demo_output")
            
            # Run evaluation
            print("\n2. Running evaluation...")
            eval_report = pipeline.run_evaluation(xml_files, "demo_evaluation.json")
            
            print(f"\nDemo completed! Check demo_output/ and demo_evaluation.json")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
