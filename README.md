# LiverTox Data Extraction Pipeline Interview Question

## Background

[LiverTox](https://www.ncbi.nlm.nih.gov/books/NBK548136/) is a clinical database maintained by the National Institute of Diabetes and Digestive and Kidney Diseases that provides information on drug-induced liver injury (DILI). Each drug entry contains structured information about hepatotoxicity patterns, risk factors, clinical presentation, and outcomes.

## Task Overview

Your goal is to build a prototype pipeline to extract structured hepatotoxicity information from LiverTox XML documents about drugs and their liver toxicity. 

You should focus on:
- Building a pipeline which can extract at least some of the fields below
- Making the pipeline robust
- Constructing evals to measure the correctness of the LLM data extraction
- Outputting a comprehensive report with these evals
- Writing good, well-structured code

This exercise is open ended: it's up to you how to implement, observe, and evaluate the pipeline. OpenAI and Anthropic API keys are provided in keys.txt for you to call into LLMs for data processing. You may install and use any other dependencies. You should use coding assistants, the web, or any other resources as much as you can to help you. 

Keep in mind limited time. You should build a minimal solution and then iterate on it. You are free to limit the scope, e.g. by starting with just extracting a subset of the desired fields. 

## Repository Structure

```
retrieval-interview/
├── livertox/                # XML files for each drug
│   ├── Acetaminophen.xml
│   ├── Warfarin.xml
│   └── ...
├── ... Your code goes here ...
├── pyproject.toml          # Project dependencies - feel free to add
└── README.md               # This file
```

### Desired Fields

The most important fields are the drug name and the DILI likelihood score: you should start with these. The likelihood score is defined at https://www.ncbi.nlm.nih.gov/books/NBK548392/ with the possible values:

- **Category A**: The drug is well known, well described and well reported to cause either direct or idiosyncratic liver injury, and has a characteristic signature; more than 50 cases including case series have been described.
- **Category B**: The drug is reported and known or highly likely to cause idiosyncratic liver injury and has a characteristic signature; between 12 and 50 cases including small case series have been described.
- **Category C**: The drug is probably linked to idiosyncratic liver injury, but has been reported uncommonly and no characteristic signature has been identified; the number of identified cases is less than 12 without significant case series.
- **Category D**: Single case reports have appeared implicating the drug, but fewer than 3 cases have been reported in the literature, no characteristic signature has been identified, and the case reports may not have been very convincing. Thus, the agent can only be said to be a possible hepatotoxin and only a rare cause of liver injury.
- **Category E**: Despite extensive use, no evidence that the drug has caused liver injury. Single case reports may have been published, but they were largely unconvincing. The agent is not believed or is unlikely to cause liver injury.
- **Category E\***: The drug is suspected to be capable of causing liver injury or idiosyncratic acute liver injury but there have been no convincing cases in the medical literature. In some situations cases of acute liver injury have been reported to regulatory agencies or mentioned in large clinical studies of the drug, but the specifics and details supportive of causality assessment are not available. The agent is unproven, but suspected to cause liver injury.
- **Category X**: Finally, for medications recently introduced into or rarely used in clinical medicine, there may be inadequate information on the risks of developing liver injury to place it in any of the five categories, and the category is characterized as “unknown.”


The full list of fields which you can consider extracting is:

- **drug_name**: Name of the medication
- **dili_likelihood_score**: LiverTox likelihood score `[A, B, C, D, E, E*, X]`
- **injury_pattern**: Type of liver injury - one of: `[hepatocellular, cholestatic, mixed, intrinsic, idiosyncratic, unclear]`
- **fraction_patients_with_enzyme_elevation**: Float (0-1), fraction of patients which had liver enzymes elevated above the upper limit of normal when taking this drug.
- **fraction_patients_with_dili**: Float (0-1), fraction of patients which had severe clinical drug-induced liver injury
- **risk_factors**: List of risk factors with supporting quotes
- **safe_dose**: Recommended safe dosage with units
- **toxic_dose**: Dose associated with toxicity
- **onset_time**: Time from exposure to injury onset
- **peak_alt**: Peak alanine aminotransferase level (IU/L)
- **peak_alp**: Peak alkaline phosphatase level (IU/L)
- **r_ratio**: ALT/ULN divided by ALP/ULN ratio
- **bilirubin_peak**: Peak bilirubin (mg/dL) for Hy's Law assessment
- **is_immune_mediated**: Boolean indicating liver injury is caused by an immune response
- **regulatory_status**: Drug status `[approved, investigational, withdrawn, unregulated]`





### Field Specifications for Data Extraction

| Field | Type | Description | Constraints/Values |
|-------|------|-------------|-------------------|
| **drug_name** | `str` | Name identifying the drug. | Non-empty string, must match XML filename (without .xml extension) |
| **dili_likelihood_score** | `enum?` | LiverTox standardized causality assessment score based on the strength of evidence linking the drug to liver injury. Derived from number of published cases, quality of evidence, and consistency of hepatotoxicity patterns. Used for clinical decision-making and regulatory assessments. | `A` (clear cause of DILI) to `E` (clearly not a cause of DILI), `X` (not enough evidence) |
| **injury_pattern** | `enum?` | A classification of how liver toxicity presents in the clinic into one of a number of categories. | `hepatocellular`, `cholestatic`, `mixed`, `intrinsic`, `idiosyncratic`, `unclear` |
| **fraction_patients_with_enzyme_elevation** | `float?` | The proportion of patients who have elevated liver enzymes when taking this drug. This represents the incidence rate in the general population or specific studied cohorts. | Range: [0.0, 1.0], nullable if data unavailable |
| **fraction_patients_with_dili** | `float?` | The proportion of patients who develop clinically significant drug-induced liver injury when taking this medication at therapeutic doses. This represents the incidence rate in the general population or specific studied cohorts. | Range: [0.0, 1.0], nullable if data unavailable |
| **risk_factors** | `List[Dict?]` | Patient-specific characteristics or conditions that increase the likelihood of developing DILI. These may include demographic factors (age, sex), genetic polymorphisms, concurrent medications, underlying diseases, or lifestyle factors. | Schema: `{factor: str, supporting_quote: str}` |
| **safe_dose** | `Dict?` | The maximum recommended therapeutic dose that can be administered without significant risk of hepatotoxicity in the general population. | Schema: `{value: float, unit: str, frequency?: str}` |
| **toxic_dose** | `Dict?` | The dose threshold above which liver injury becomes likely or has been observed. | Schema: `{value: float, unit: str, frequency?: str}` |
| **onset_time** | `Dict?` | The typical time range from exposure to injury onset. | Schema: `{min?: int, max?: int, typical?: int, unit: str}` |
| **peak_alt** | `float?` | Maximum observed level of alanine aminotransferase (a liver enzyme) during a typical DILI episode. | Reported in multiples of the upper limit of normal (ULN). |
| **peak_alp** | `float?` | Maximum observed level of alkaline phosphatase (a liver enzyme) during a typical DILI episode. | Reported in multiples of the upper limit of normal (ULN). |
| **r_ratio** | `float?` | A calculated value that helps classify the pattern of liver injury by comparing the degree of ALT vs ALP elevation relative to their upper limits of normal: i.e. "ALT elevation" / "ALP elevation" where both elevations are reported as multiples of the upper limit of normal. | >5: hepatocellular, <2: cholestatic, 2-5: mixed |
| **bilirubin_peak** | `float?` | Maximum observed level of bilirubin (a liver product) during a typical DILI episode. | Reported in multiples of the upper limit of normal (ULN). |
| **is_immune_mediated** | `bool` | Indicates whether the liver toxicity is caused by a response of the immune system to the drug or its effects. Characterized by systemic hypersensitivity features (fever, rash, eosinophilia), presence of autoantibodies, and often longer latency period. Affects treatment approach and rechallenge decisions. | Default: `false` |
| **regulatory_status** | `enum?` | Current regulatory approval status reflecting the drug's availability for clinical use. | `approved`, `investigational`, `withdrawn`, `unregulated` |

**Notes:**
- `?` indicates nullable/optional fields: all fields are optional except for `drug_name`
- ULN = Upper Limit of Normal
- Doses should preserve original units from source

## Resources

- [LiverTox Database](https://www.ncbi.nlm.nih.gov/books/NBK547852/)
- [Drug-Induced Liver Injury Overview](https://www.ncbi.nlm.nih.gov/books/NBK548136/)
