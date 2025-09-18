#!/usr/bin/env python3
"""
Demonstration script for multi-model support (OpenAI and Anthropic).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline import HepatotoxicityPipeline
from src.config import OPENAI_MODELS, ANTHROPIC_MODELS


def main():
    """Demonstrate multi-model support."""
    
    print("🤖 MULTI-MODEL HEPATOTOXICITY EXTRACTION DEMONSTRATION")
    print("=" * 80)
    
    print("🔧 SUPPORTED MODELS:")
    print("\nOpenAI Models:")
    for model in OPENAI_MODELS:
        print(f"  • {model}")
    
    print("\nAnthropic Models:")
    for model in ANTHROPIC_MODELS:
        print(f"  • {model}")
    
    print("\n📋 USAGE EXAMPLES:")
    print("-" * 80)
    
    print("1. Use Claude 3.5 Sonnet for both retriever and evaluator:")
    print("   python -m src.pipeline extract --drug Acetaminophen --model claude-3-5-sonnet-20241022")
    print("   python -m src.pipeline evaluate --drug Acetaminophen --model claude-3-5-sonnet-20241022")
    
    print("\n2. Use different models for retriever vs evaluator:")
    print("   python -m src.pipeline evaluate --drug Acetaminophen \\")
    print("     --retriever-model gpt-4o --evaluator-model claude-3-5-sonnet-20241022")
    
    print("\n3. Use GPT-4 Turbo:")
    print("   python -m src.pipeline extract --drug Buspirone --model gpt-4-turbo")
    
    print("\n4. Compare models (extract with different models):")
    print("   python -m src.pipeline extract --drug Acetaminophen --model gpt-4o --output gpt4_result.json")
    print("   python -m src.pipeline extract --drug Acetaminophen --model claude-3-5-sonnet-20241022 --output claude_result.json")
    
    # Quick test if we have the drugs available
    try:
        pipeline = HepatotoxicityPipeline()
        xml_files = pipeline.get_available_drugs()
        
        if xml_files:
            test_file = xml_files[0]
            drug_name = Path(test_file).stem
            
            print(f"\n🧪 QUICK MODEL TEST (using {drug_name}):")
            print("-" * 80)
            
            # Test with default model (GPT-4o)
            print("Testing with default model (GPT-4o)...")
            pipeline_gpt = HepatotoxicityPipeline()
            result_gpt = pipeline_gpt.extract_single_drug(test_file)
            
            print(f"✅ GPT-4o extraction completed")
            print(f"   Drug: {result_gpt['extraction_result']['data']['drug_name']}")
            print(f"   DILI Score: {result_gpt['extraction_result']['data'].get('dili_likelihood_score', 'None')}")
            
            # Test with Claude if available (you can uncomment to test)
            # print("\nTesting with Claude 3.5 Sonnet...")
            # pipeline_claude = HepatotoxicityPipeline(
            #     retriever_model="claude-3-5-sonnet-20241022"
            # )
            # result_claude = pipeline_claude.extract_single_drug(test_file)
            # print(f"✅ Claude extraction completed")
            # print(f"   Drug: {result_claude['extraction_result']['data']['drug_name']}")
            # print(f"   DILI Score: {result_claude['extraction_result']['data'].get('dili_likelihood_score', 'None')}")
            
        else:
            print("\n⚠️  No XML files found for testing")
    
    except Exception as e:
        print(f"\n⚠️  Test failed: {e}")
    
    print(f"\n{'MULTI-MODEL CONFIGURATION BENEFITS':^80}")
    print("=" * 80)
    print("✅ Cost Optimization: Use different models based on task complexity")
    print("✅ Performance Comparison: Test multiple models on same data")
    print("✅ Provider Flexibility: Switch between OpenAI and Anthropic")
    print("✅ Specialized Models: Use Claude for reasoning, GPT for structured output")
    print("✅ Fallback Options: Switch providers if one is unavailable")
    
    print(f"\n{'CONFIGURATION HIERARCHY':^80}")
    print("-" * 80)
    print("1. --model: Sets both retriever and evaluator to same model")
    print("2. --retriever-model: Overrides model for retriever only")
    print("3. --evaluator-model: Overrides model for evaluator only")
    print("4. Default: Uses configuration from src/config.py")
    
    print("\n" + "=" * 80)
    print("🎉 Multi-model support ready! Choose your preferred models.")
    print("=" * 80)


if __name__ == "__main__":
    main()
