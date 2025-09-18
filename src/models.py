"""
Pydantic models for structured hepatotoxicity data extraction.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator


class RiskFactor(BaseModel):
    """Risk factor with supporting evidence."""
    factor: str = Field(..., description="The risk factor")
    supporting_quote: str = Field(..., description="Supporting quote from the document")


class Dose(BaseModel):
    """Dose information with value, unit, and optional frequency."""
    value: float = Field(..., description="Numeric dose value")
    unit: str = Field(..., description="Dose unit (e.g., mg, g)")
    frequency: Optional[str] = Field(None, description="Frequency of administration")


class OnsetTime(BaseModel):
    """Time from exposure to injury onset."""
    min: Optional[int] = Field(None, description="Minimum time")
    max: Optional[int] = Field(None, description="Maximum time")
    typical: Optional[int] = Field(None, description="Typical time")
    unit: str = Field(..., description="Time unit (e.g., days, hours, weeks)")


class HepatotoxicityData(BaseModel):
    """Complete hepatotoxicity data structure for a drug."""
    
    # Required field
    drug_name: str = Field(..., description="Name of the medication")
    
    # DILI likelihood score - most important after drug_name
    dili_likelihood_score: Optional[Literal["A", "B", "C", "D", "E", "E*", "X"]] = Field(
        None, description="LiverTox likelihood score"
    )
    
    # Other optional fields
    injury_pattern: Optional[Literal[
        "hepatocellular", "cholestatic", "mixed", "intrinsic", "idiosyncratic", "unclear"
    ]] = Field(None, description="Type of liver injury")
    
    fraction_patients_with_enzyme_elevation: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of patients with elevated liver enzymes"
    )
    
    fraction_patients_with_dili: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of patients with severe DILI"
    )
    
    risk_factors: Optional[List[RiskFactor]] = Field(
        None, description="List of risk factors with supporting quotes"
    )
    
    safe_dose: Optional[Dose] = Field(None, description="Recommended safe dosage")
    toxic_dose: Optional[Dose] = Field(None, description="Dose associated with toxicity")
    onset_time: Optional[OnsetTime] = Field(None, description="Time from exposure to injury onset")
    
    peak_alt: Optional[float] = Field(None, description="Peak ALT level (multiples of ULN)")
    peak_alp: Optional[float] = Field(None, description="Peak ALP level (multiples of ULN)")
    r_ratio: Optional[float] = Field(None, description="ALT/ULN divided by ALP/ULN ratio")
    bilirubin_peak: Optional[float] = Field(None, description="Peak bilirubin (multiples of ULN)")
    
    is_immune_mediated: Optional[bool] = Field(False, description="Whether liver injury is immune-mediated")
    
    regulatory_status: Optional[Literal[
        "approved", "investigational", "withdrawn", "unregulated"
    ]] = Field(None, description="Drug regulatory status")
    
    # Source information for traceability
    extraction_sources: Optional[Dict[str, str]] = Field(
        None, description="Sources for each extracted field"
    )


class ExtractionResult(BaseModel):
    """Result of data extraction including metadata."""
    data: HepatotoxicityData
    extraction_timestamp: str
    model_used: str
    confidence_score: Optional[float] = None
    extraction_notes: Optional[str] = None


class EvaluationQuestion(BaseModel):
    """A question for evaluating the retriever."""
    question: str
    expected_fields: List[str]
    drug_name: str


class EvaluationResult(BaseModel):
    """Result of evaluating a retriever response."""
    question: str
    drug_name: str
    retriever_response: HepatotoxicityData
    correctness_scores: Dict[str, float]  # field -> score (0-1)
    overall_score: float
    feedback: str
    hallucination_detected: bool
    missing_information: List[str]
