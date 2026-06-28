import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from groq import Groq

load_dotenv()

# ----------------------------------------------------------------------
# API Key — read once from the environment only. Never collected or
# displayed in the UI, never written back to disk.
# ----------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def has_api_key() -> bool:
    return bool(GROQ_API_KEY)


def _require_key() -> str:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set in the environment. Add it to your .env file "
            "or export it before starting the app."
        )
    return GROQ_API_KEY


# ----------------------------------------------------------------------
# Pydantic Schemas for Structured Outputs
# ----------------------------------------------------------------------
class IntentAndLanguage(BaseModel):
    # Internal classification only — never surfaced in the UI.
    mode: Literal["A", "B", "C", "D"] = Field(
        description="Classification string: A, B, C, or D based on specifications."
    )
    source_lang: str = Field(
        description="2-letter ISO language code detected from input (e.g., 'en', 'ta', 'hi')."
    )
    target_lang: str = Field(
        description="Target 2-letter ISO language code for output (e.g., 'en', 'te')."
    )


class DetectedLanguage(BaseModel):
    source_lang: str = Field(
        description="2-letter ISO code of the input text's language, e.g. 'en', 'ta', 'hi'."
    )


class TextSegment(BaseModel):
    id: int
    text: str
    lang: str = Field(description="'en' or the detected regional language code.")
    voice: str = Field(description="Voice matching the Voice Routing Table specification.")
    pause_after: Optional[str] = Field(None, description="PAUSE_SHORT or PAUSE_LONG if applicable.")


class WordTimestamp(BaseModel):
    word: str
    start_ms: int
    end_ms: int
    seg_id: int


# NOTE: original_text / translated_text are intentionally NOT part of this
# schema. Letting the LLM restate them caused it to sometimes echo the
# original text back as the "translation". Both fields are injected
# directly from already-computed pipeline values after this call returns.
class SegmentationOutput(BaseModel):
    segments: List[TextSegment]
    word_timestamps: List[WordTimestamp]


# ----------------------------------------------------------------------
# LangGraph State Definition
# ----------------------------------------------------------------------
class AgentState(BaseModel):
    input_text: str = ""
    context_document: str = ""
    target_lang_override: Optional[str] = None

    # Internal states (never surfaced to the UI)
    mode: str = "A"
    source_lang: str = "en"
    target_lang: str = "en"
    processed_text: str = ""
    romanized_text: str = ""

    # Final Structured Output
    final_payload: Optional[Dict[str, Any]] = None


# ----------------------------------------------------------------------
# Voice Routing Table Lookup Helper
# ----------------------------------------------------------------------
VOICE_ROUTING = {
    "ta": "ta-IN-PallaviNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "pa": "pa-IN-OjasNeural",
    "ur": "ur-PK-UzmaNeural",
    "en": "en-IN-NeerjaNeural",
}


def _make_llm(temperature: float = 0.3) -> ChatGroq:
    key = _require_key()
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature, api_key=key)


def _groq_client() -> Groq:
    key = _require_key()
    return Groq(api_key=key)


# ----------------------------------------------------------------------
# Graph Nodes
# ----------------------------------------------------------------------
def detect_intent_and_language(state: AgentState) -> Dict[str, Any]:
    llm = _make_llm(temperature=0.3).with_structured_output(IntentAndLanguage)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Intent and Language recognition module of VoiceBridge.
        Analyze the input and classify into:
        - MODE A: Translation/Statements (no clear question markers).
        - MODE B: Audio handling (Triggered if context hints audio source).
        - MODE C: Q&A (Contains explicit or implicit questions, ?, what, how, why, etc.).
        - MODE D: Contextual question about uploaded document/audio.

        Identify the source language (en, ta, hi, te, kn, ml, bn, mr, gu, pa, or, ur).
        Determine the logical target language (If input is english -> match chosen target or default regional; if input is regional -> english).
        """),
        ("human", "Input text: {input_text}\nContext available: {context_document}")
    ])

    chain = prompt | llm
    res = chain.invoke({"input_text": state.input_text, "context_document": state.context_document})

    target = state.target_lang_override if state.target_lang_override else res.target_lang
    if target == res.source_lang and res.source_lang == "en":
        target = "ta"  # Fallback variant mapping

    return {
        "mode": res.mode,
        "source_lang": res.source_lang,
        "target_lang": target,
    }


def process_core_logic(state: AgentState) -> Dict[str, Any]:
    llm = _make_llm(temperature=0.3)

    if state.mode in ["C", "D"]:
        system_prompt = f"""You are VoiceBridge Q&A engine. Answer the question comprehensively in 2-4 sentences.
        The final output response MUST be entirely written in the target language code: '{state.target_lang}'.
        Context document if any: {state.context_document}"""
    else:
        system_prompt = f"""You are VoiceBridge Translation engine. Translate the incoming text explicitly into the target language: '{state.target_lang}'.
        Retain popular English loan words (like delivery, address, account, Amazon, online, OTP) in English characters directly within the translation text to support code-switching logic downstream."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input_text}")
    ])

    chain = prompt | llm
    res = chain.invoke({"input_text": state.input_text})
    processed_text = str(res.content).strip()

    roman_prompt = (
        "Convert the following text into Roman script representation "
        f"(Transliteration) for users who cannot read the native script. Text: {processed_text}"
    )
    roman_res = llm.invoke(roman_prompt)

    return {"processed_text": processed_text, "romanized_text": str(roman_res.content).strip()}


def segment_and_structure(state: AgentState) -> Dict[str, Any]:
    llm = _make_llm(temperature=0).with_structured_output(SegmentationOutput)

    system_prompt = f"""You are VoiceBridge Audio Segmentation engine. Take the PROCESSED OUTPUT TEXT
    (never the original input text) and structure it exactly into segments matching these rules:
    - Rule 1: Segments should be multi-word phrases (at least 2 words when possible).
    - Rule 2/3: Brand names (Amazon, Zomato) and English loan words (delivery, address, online) must be explicit segments with lang='en' and voice='en-IN-NeerjaNeural'.
    - Rule 4: Numbers and units keep as English segments.
    - Rule 6: Assign PAUSE_SHORT after commas, and PAUSE_LONG after sentence completions.

    Generate synthetic logical monotonic word_timestamps matching the reading flow of sentences.
    Only segment the processed output text below - do not segment or return the original input text.

    Primary target voice for non-English segments: {VOICE_ROUTING.get(state.target_lang, "en-IN-NeerjaNeural")}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Processed output text to segment: {proc}")
    ])

    chain = prompt | llm
    structured_res = chain.invoke({"proc": state.processed_text})

    # original_text and translated_text are injected here directly from
    # pipeline state - the LLM never gets a chance to restate or swap them.
    final_payload = {
        "session_id": "vb_local",
        "mode": state.mode,
        "source_lang": state.source_lang,
        "target_lang": state.target_lang,
        "original_text": state.input_text,
        "translated_text": state.processed_text,
        "romanized_text": state.romanized_text,
        "code_switched": any(seg.lang == "en" for seg in structured_res.segments) and state.target_lang != "en",
        "segments": [seg.model_dump() for seg in structured_res.segments],
        "word_timestamps": [wt.model_dump() for wt in structured_res.word_timestamps],
    }

    return {"final_payload": final_payload}


# ----------------------------------------------------------------------
# Graph Generation
# ----------------------------------------------------------------------
def build_voicebridge_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("detector", detect_intent_and_language)
    workflow.add_node("processor", process_core_logic)
    workflow.add_node("segmenter", segment_and_structure)

    workflow.set_entry_point("detector")
    workflow.add_edge("detector", "processor")
    workflow.add_edge("processor", "segmenter")
    workflow.add_edge("segmenter", END)

    return workflow.compile()


voicebridge_agent = build_voicebridge_graph()


# ----------------------------------------------------------------------
# Lightweight helper functions used directly by the Streamlit UI for
# Mode A (text) and Mode B (voice) flows, plus follow-up Q&A and the
# on-demand "Explain" action. These sit alongside the LangGraph pipeline
# above rather than replacing it.
# ----------------------------------------------------------------------
def _detect_source_lang(text: str) -> str:
    llm = _make_llm(temperature=0).with_structured_output(DetectedLanguage)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Identify the dominant language of the given text. Return only a 2-letter ISO code "
                   "(en, ta, hi, te, kn, ml, bn, mr, gu, pa, or, ur)."),
        ("human", "{text}")
    ])
    res = (prompt | llm).invoke({"text": text})
    return res.source_lang


def translate_text(input_text: str, target_lang: Optional[str]) -> Dict[str, Any]:
    """Core Mode A / Mode B conversion step used by the UI: takes raw text
    and turns it into the desired target language directly, without going
    through the full intent-classification graph."""
    source_lang = _detect_source_lang(input_text)
    target = target_lang or ("en" if source_lang != "en" else "ta")

    llm = _make_llm(temperature=0.3)
    system_prompt = f"""You are VoiceBridge Translation engine. Translate the incoming text explicitly into the
    target language: '{target}'. Retain popular English loan words (like delivery, address, account, Amazon,
    online, OTP) in English characters directly within the translation text to support code-switching."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input_text}")
    ])
    res = (prompt | llm).invoke({"input_text": input_text})
    translated_text = str(res.content).strip()

    roman_prompt = (
        "Convert the following text into Roman script representation (transliteration) for users "
        f"who cannot read the native script. Text: {translated_text}"
    )
    roman_res = llm.invoke(roman_prompt)
    romanized_text = str(roman_res.content).strip()

    return {
        "source_lang": source_lang,
        "target_lang": target,
        "original_text": input_text,
        "translated_text": translated_text,
        "romanized_text": romanized_text,
    }


def answer_followup_question(context_text: str, question: str, target_lang: str) -> str:
    """Used for the single continuation question that appears after a
    translation in both Mode A and Mode B."""
    llm = _make_llm(temperature=0.3)
    system_prompt = f"""You are VoiceBridge's follow-up assistant. The user previously converted/translated the
    text below, and is now asking a question about it. Answer the question clearly in 2-4 sentences, written
    entirely in the target language code: '{target_lang}'.

    Converted text context: {context_text}"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    res = (prompt | llm).invoke({"question": question})
    return str(res.content).strip()


def explain_text(text: str, target_lang: str) -> str:
    """Only called when the user clicks the 'Explain' button — nothing is
    generated until then."""
    llm = _make_llm(temperature=0.3)
    system_prompt = f"""Explain the meaning and context of the following text in simple terms, in 3-5 sentences,
    written entirely in the target language code: '{target_lang}'."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])
    res = (prompt | llm).invoke({"text": text})
    return str(res.content).strip()


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Mode B: turn an uploaded or recorded voice clip into text using
    Groq's hosted Whisper model."""
    client = _groq_client()
    transcript = client.audio.transcriptions.create(
        file=(filename, audio_bytes),
        model="whisper-large-v3",
    )
    return transcript.text.strip()


def build_segments_for_audio(translated_text: str, target_lang: str) -> List[Dict[str, Any]]:
    """Lightweight segmentation purely for TTS language tagging in the UI
    (no LLM call needed for this simple step)."""
    return [{
        "text": translated_text,
        "lang": target_lang,
        "voice": VOICE_ROUTING.get(target_lang, "en-IN-NeerjaNeural"),
    }]