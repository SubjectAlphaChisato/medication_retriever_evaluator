#!/usr/bin/env python3
"""
Test script for the hepatotoxicity extraction pipeline.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import HepatotoxicityPipeline
from src.retriever import HepatotoxicityRetriever
from src.xml_parser import LiverToxXMLParser


def test_xml_parser():
    """Test XML parsing functionality."""
    print("Testing XML Parser...")
    
    # Test with Acetaminophen
    xml_file = "livertox/Acetaminophen.xml"
    if not Path(xml_file).exists():
        print(f"Error: {xml_file} not found")
        return False
    
    try:
        parser = LiverToxXMLParser(xml_file)
        drug_name = parser.get_drug_name()
        print(f"Drug name: {drug_name}")
        
        # Test section extraction
        sections = parser.extract_sections()
        print(f"Found {len(sections)} sections")
        
        # Test likelihood score detection
        likelihood_context = parser.find_likelihood_score()
        if likelihood_context:
            print(f"Likelihood context found: {likelihood_context[:100]}...")
        
        print("✓ XML Parser test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ XML Parser test failed: {e}\n")
        return False


def test_retriever():
    """Test retriever functionality."""
    print("Testing Retriever...")
    
    try:
        retriever = HepatotoxicityRetriever()
        
        # Test extraction
        xml_file = "livertox/Buspirone.xml"  # Smaller file for testing
        if not Path(xml_file).exists():
            print(f"Error: {xml_file} not found")
            return False
        
        result = retriever.extract_hepatotoxicity_data(xml_file)
        
        print(f"Extracted data for: {result.data.drug_name}")
        print(f"DILI likelihood score: {result.data.dili_likelihood_score}")
        print(f"Injury pattern: {result.data.injury_pattern}")
        print(f"Model used: {result.model_used}")
        
        print("✓ Retriever test passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Retriever test failed: {e}\n")
        return False


def test_demo():
    """Run a quick demo of the pipeline."""
    print("Running Demo...")
    
    try:
        pipeline = HepatotoxicityPipeline()
        
        # Get first available XML file
        xml_files = pipeline.get_available_drugs()
        if not xml_files:
            print("No XML files found")
            return False
        
        # Test single extraction
        test_file = xml_files[0]  # Use first file
        drug_name = Path(test_file).stem
        
        print(f"Testing extraction for {drug_name}...")
        
        result = pipeline.extract_single_drug(test_file)
        data = result["extraction_result"]["data"]
        
        print(f"\nExtraction Results for {data['drug_name']}:")
        print(f"  DILI likelihood score: {data.get('dili_likelihood_score', 'Unknown')}")
        print(f"  Injury pattern: {data.get('injury_pattern', 'Unknown')}")
        print(f"  Is immune mediated: {data.get('is_immune_mediated', False)}")
        print(f"  Regulatory status: {data.get('regulatory_status', 'Unknown')}")
        
        # Show sources if available
        sources = data.get('extraction_sources', {})
        if sources:
            print("\nExtraction Sources:")
            for field, source in sources.items():
                print(f"  {field}: {source[:50]}...")
        
        print("\n✓ Demo test passed")
        return True
        
    except Exception as e:
        print(f"✗ Demo test failed: {e}")
        return False


def test_evaluation():
    """Test the evaluation system."""
    print("Testing Evaluation System...")
    
    try:
        pipeline = HepatotoxicityPipeline()
        
        # Get first XML file for evaluation test
        xml_files = pipeline.get_available_drugs()
        if not xml_files:
            print("No XML files found")
            return False
        
        # Test evaluation on just one drug
        test_files = xml_files[:1]  # Just first file
        drug_name = Path(test_files[0]).stem
        
        print(f"Testing evaluation for {drug_name}...")
        
        # Run evaluation with specific question types
        report = pipeline.run_evaluation(
            test_files, 
            f"test_evaluation_{drug_name}.json",
            question_types=["likelihood_assessment", "clinical_patterns"]  # Test subset
        )
        
        # Check that we got a valid report
        if "summary_statistics" in report and "detailed_results" in report:
            stats = report["summary_statistics"]
            print(f"✓ Evaluation completed:")
            print(f"  - Drugs evaluated: {stats.get('total_drugs', 0)}")
            print(f"  - Questions asked: {stats.get('total_questions', 0)}")
            print(f"  - Average score: {stats.get('average_overall_score', 0):.2f}")
            return True
        else:
            print("✗ Invalid evaluation report structure")
            return False
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HEPATOTOXICITY EXTRACTION PIPELINE TESTS")
    print("=" * 60)
    
    # Check if data directory exists
    if not Path("livertox").exists():
        print("Error: livertox directory not found")
        print("Please ensure the XML files are in the livertox/ directory")
        return
    
    # Run tests
    tests = [
        test_xml_parser,
        test_retriever,
        test_demo,
        test_evaluation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\nTo run the full pipeline:")
        print("  python -m src.pipeline demo")
        print("  python -m src.pipeline extract --drug Acetaminophen")
        print("  python -m src.pipeline evaluate --limit 3")
    else:
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
