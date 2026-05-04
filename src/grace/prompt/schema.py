# schema.py
# ----------------------------------------------------------------------------------------
# structured json scheme outputs for LLM response formats
# ----------------------------------------------------------------------------------------
# adriana r.f. (@adrmisty:github, arodriguezf@vicomtech.org)
# apr-2026

from pydantic import BaseModel, Field
from typing import List, Literal

class Premise(BaseModel):
    local_id: str = Field(description="Local identifier for the premise, e.g., 'p1', 'p2'.")
    source_index: int = Field(description="Index of the sentence where the premise was found in the text")
    text: str = Field(description="Exact minimal text fragment representing the clinical premise")

class Claim(BaseModel):
    id: str = Field(description="ID of the option/claim, e.g., '1', '2'")
    text: str = Field(description="Exact text of the option/claim")

class Relation(BaseModel):
    premise_id: str = Field(description="ID of the premise (must exist in premises list)")
    claim_id: str = Field(description="ID of the claim (must correspond to an option)")
    relation_type: Literal["Support", "Attack"] = Field(description="Type of relation: 'Support' or 'Attack'")

class SchemaS1(BaseModel):
    sentence_relevancy: List[Literal["relevant", "not-relevant"]] = Field(
        description="List of relevancy labels for each sentence in the clinical case"
    )

class SchemaS2(BaseModel):
    premises: List[Premise] = Field(description="List of extracted minimal premises")
    claims: List[Claim] = Field(description="List of extracted claims/options")

class SchemaS3(BaseModel):
    relations: List[Relation] = Field(description="List of argumentative relations between premises and claims")
    
class SchemaGlobal(BaseModel):
    sentence_relevancy: List[Literal["relevant", "not-relevant"]] = Field(
        description="List of relevancy labels for each sentence in the clinical case."
    )
    premises: List[Premise] = Field(description="List of extracted minimal premises.")
    relations: List[Relation] = Field(description="List of argumentative relations between premises and claims.")