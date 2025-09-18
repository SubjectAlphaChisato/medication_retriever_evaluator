"""
XML parsing utilities for LiverTox documents.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from pathlib import Path


class LiverToxXMLParser:
    """Parser for LiverTox XML documents."""
    
    def __init__(self, xml_file_path: str):
        """Initialize parser with XML file path."""
        self.xml_file_path = Path(xml_file_path)
        self.tree = None
        self.root = None
        self._parse_xml()
    
    def _parse_xml(self) -> None:
        """Parse the XML file."""
        try:
            self.tree = ET.parse(self.xml_file_path)
            self.root = self.tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML file {self.xml_file_path}: {e}")
    
    def get_drug_name(self) -> str:
        """Extract drug name from XML title."""
        # Get the title from the XML document
        title_elem = self.root.find(".//title")
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()
        
        # Fallback to filename if title not found
        return self.xml_file_path.stem
    
    def extract_text_content(self) -> str:
        """Extract all text content from the XML document."""
        def extract_text_recursive(element):
            text_content = []
            if element.text:
                text_content.append(element.text.strip())
            
            for child in element:
                text_content.extend(extract_text_recursive(child))
                if child.tail:
                    text_content.append(child.tail.strip())
            
            return [t for t in text_content if t]
        
        all_text = extract_text_recursive(self.root)
        return " ".join(all_text)
    
    def extract_sections(self) -> Dict[str, str]:
        """Extract content organized by sections."""
        sections = {}
        
        # Find all section elements
        for sec in self.root.findall(".//sec"):
            sec_id = sec.get("id", "")
            
            # Get section title
            title_elem = sec.find("title")
            title = title_elem.text.strip() if title_elem is not None else "Untitled"
            
            # Extract text content from this section only (not subsections)
            section_text = []
            for elem in sec:
                if elem.tag == "p":
                    if elem.text:
                        section_text.append(elem.text.strip())
                elif elem.tag == "title":
                    continue  # Already got the title
                # Add other relevant elements as needed
            
            if section_text:
                sections[f"{title} ({sec_id})"] = " ".join(section_text)
        
        return sections
    
    def find_likelihood_score(self) -> Optional[str]:
        """Try to find likelihood score in the document."""
        text_content = self.extract_text_content().lower()
        
        # Look for likelihood score patterns
        score_patterns = [
            "likelihood score:",
            "likelihood:",
            "category a",
            "category b", 
            "category c",
            "category d",
            "category e",
            "category e*",
            "category x"
        ]
        
        for pattern in score_patterns:
            if pattern in text_content:
                # Extract surrounding text for context
                start_idx = text_content.find(pattern)
                context = text_content[max(0, start_idx-50):start_idx+200]
                return context
        
        return None
    
    def find_injury_pattern(self) -> Optional[str]:
        """Try to find injury pattern information."""
        text_content = self.extract_text_content().lower()
        
        patterns = [
            "hepatocellular",
            "cholestatic", 
            "mixed",
            "intrinsic",
            "idiosyncratic"
        ]
        
        found_patterns = []
        for pattern in patterns:
            if pattern in text_content:
                found_patterns.append(pattern)
        
        return ", ".join(found_patterns) if found_patterns else None
    
    def extract_case_reports(self) -> List[Dict[str, str]]:
        """Extract case report sections."""
        case_reports = []
        
        # Look for case report sections
        for sec in self.root.findall(".//sec"):
            sec_id = sec.get("id", "")
            title_elem = sec.find("title")
            title = title_elem.text.strip() if title_elem is not None else ""
            
            if "case" in title.lower():
                # Extract case content
                case_text = []
                for p in sec.findall(".//p"):
                    if p.text:
                        case_text.append(p.text.strip())
                
                case_reports.append({
                    "title": title,
                    "id": sec_id,
                    "content": " ".join(case_text)
                })
        
        return case_reports
