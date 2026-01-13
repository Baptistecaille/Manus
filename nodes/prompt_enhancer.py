"""
Prompt Enhancer Node - Context Engineering for Manus Agent.

This node analyzes user queries, enriches them with context,
detects risks, and prepares structured input for the Planner.
"""

import json
import logging
import os
import re
from typing import Optional

from pydantic import BaseModel, Field

from agent_state import AgentStateDict, estimate_tokens
from llm_factory import create_llm
from seedbox_executor import SeedboxExecutor
from nodes.schema import (
    EnhancerOutput,
    AnalysisResult,
    ContextEnrichment,
    RiskAssessment,
    RiskFactor,
    ExecutionHints,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVE KEYWORDS FOR RISK DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

SENSITIVE_KEYWORDS = {
    "critical": [
        "rm -rf /",
        "rm -rf /*",
        "dd if=/dev/zero",
        "mkfs",
        ":(){ :|:& };:",  # Fork bomb
        "> /dev/sda",
        "chmod -R 777 /",
        "sudo rm",
    ],
    "high": [
        "rm -rf",
        "rm -r",
        "sudo",
        "chmod 777",
        "curl | bash",
        "curl | sh",
        "wget | bash",
        "wget | sh",
        "eval",
        "exec",
        "/etc/passwd",
        "/etc/shadow",
        "dd if=",
    ],
    "medium": [
        "rm",
        ">",  # Redirect (overwrite)
        ">>",  # Redirect (append)
        "chmod",
        "chown",
        "/etc/",
        "/sys/",
        "/proc/",
        "kill",
        "pkill",
        "systemctl",
        "service",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

# Validation models are now imported from nodes.schema


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

from prompts import get_base_system_prompt

# Base capabilities from centralized source
BASE_CAPABILITIES = get_base_system_prompt()

SYSTEM_PROMPT = f"""# RÔLE
Tu es le "Context Engineering Module" de Manus Agent, un système d'analyse
et d'optimisation de requêtes pour agents autonomes.

# CONTEXTE DE L'AGENT
{BASE_CAPABILITIES}

# MISSION
Transformer une requête utilisateur brute en une spécification structurée
et contextualisée pour permettre une exécution fiable et sécurisée.

# PROCESSUS D'ANALYSE

## ÉTAPE 1: Décomposition de l'Intention
Analyse la requête selon ces dimensions:
- **Intent Principal**: Quelle est l'action finale attendue?
- **Sous-intents**: Quelles étapes intermédiaires sont nécessaires?
- **Contraintes Explicites**: Limitations mentionnées par l'utilisateur
- **Contraintes Implicites**: Attentes non formulées mais évidentes

## ÉTAPE 2: Classification du Domaine
Catégorise la requête:
- code_generation: Écriture/modification de code
- web_research: Recherche d'informations en ligne
- data_analysis: Traitement et analyse de données
- file_manipulation: Opérations sur le système de fichiers
- mixed_workflow: Combinaison de plusieurs domaines

Fournis un score de confiance (0.0-1.0).

## ÉTAPE 3: Injection de Contexte
Les informations de contexte (fichiers, contraintes) sont fournies dynamiquement ci-dessous.
Intègre-les pour vérifier la faisabilité et la sécurité.

## ÉTAPE 4: Détection des Risques
Identifie les signaux de danger:

### 🔴 RISQUE CRITIQUE
- Commandes destructives (rm -rf, dd, mkfs)
- Modification de configurations système
- Accès root/sudo requis

### 🟡 RISQUE MODÉRÉ
- Boucles infinies potentielles
- Opérations réseau non vérifiées
- Modifications de fichiers sans backup

### 🟢 RISQUE FAIBLE
- Lecture seule de fichiers
- Recherches web standards
- Génération de code sans exécution

## ÉTAPE 5: Reformulation Optimisée
Produis une version améliorée qui:
1. Élimine toute ambiguïté
2. Spécifie les outputs attendus (format, structure)
3. Intègre les contraintes de l'environnement
4. Ajoute des critères de validation du succès

# FORMAT DE SORTIE
# Tu DOIS formater ta réponse selon le schéma JSON EnhancerOutput.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def detect_keyword_risk(text: str) -> tuple[str, list[dict]]:
    """
    Detect sensitive keywords in text and return risk level.

    Args:
        text: Text to analyze.

    Returns:
        Tuple of (risk_level, list of risk_factors).
    """
    text_lower = text.lower()
    risk_factors = []
    max_level = "low"

    level_priority = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    for level, keywords in SENSITIVE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                risk_factors.append(
                    {
                        "type": f"sensitive_keyword_{level}",
                        "severity": {"critical": 10, "high": 8, "medium": 5, "low": 2}[
                            level
                        ],
                        "description": f"Detected sensitive keyword: '{keyword}'",
                    }
                )
                if level_priority[level] > level_priority[max_level]:
                    max_level = level

    return max_level, risk_factors


def collect_workspace_context() -> dict:
    """
    Collect workspace context from the seedbox.

    Returns:
        Dict with workspace information.
    """
    try:
        executor = SeedboxExecutor()

        # Get file listing
        files = executor.list_files("/workspace")

        # Get current directory structure (limited depth)
        result = executor.execute_bash("find /workspace -maxdepth 2 -type f | head -20")
        file_tree = (
            result.get("stdout", "").strip().split("\n")
            if result.get("success")
            else []
        )

        return {
            "current_directory": "/workspace",
            "file_tree": file_tree,
            "files_count": len(files),
            "available_tools": [
                "bash",
                "python",
                "search",
                "playwright",
                "deep_research",
            ],
        }
    except Exception as e:
        logger.warning(f"Failed to collect workspace context: {e}")
        return {
            "current_directory": "/workspace",
            "file_tree": [],
            "files_count": 0,
            "available_tools": [
                "bash",
                "python",
                "search",
                "playwright",
                "deep_research",
            ],
            "error": str(e),
        }


# Helper function `parse_enhancer_response` is no longer needed but kept empty to avoid breaking refs if any
def parse_enhancer_response(response_text: str) -> Optional[EnhancerOutput]:
    """Deprecated: Parsing is now handled by with_structured_output."""
    return None


def determine_hitl_mode(risk_level: str, current_mode: str) -> str:
    """
    Determine the HITL mode based on risk level.

    Args:
        risk_level: Detected risk level.
        current_mode: Currently configured HITL mode.

    Returns:
        Final HITL mode to use.
    """
    # Risk level can override to stricter mode, but not to more lenient
    risk_to_hitl = {
        "critical": "strict",
        "high": "strict",
        "medium": "moderate",
        "low": "minimal",
    }

    mode_priority = {"minimal": 0, "moderate": 1, "strict": 2}

    risk_suggested = risk_to_hitl.get(risk_level, "moderate")

    # Use the stricter of the two
    if mode_priority.get(risk_suggested, 1) > mode_priority.get(current_mode, 1):
        return risk_suggested
    return current_mode


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN NODE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def prompt_enhancer_node(state: AgentStateDict) -> dict:
    """
    LangGraph node that enhances the user query with context and risk analysis.

    This node:
    1. Collects workspace context
    2. Calls LLM to analyze and enhance the query
    3. Detects risks via keyword analysis
    4. Sets appropriate HITL mode
    5. Triggers first breakpoint for validation

    Args:
        state: Current agent state.

    Returns:
        Dict of state updates.
    """
    logger.info("Prompt Enhancer - Starting query analysis")

    original_query = state.get("original_query", "")
    if not original_query:
        # Fallback: extract from first user message
        messages = state.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                original_query = msg.get("content", "")
                break

    # Collect workspace context
    workspace_context = collect_workspace_context()

    # Get technical constraints
    constraints = state.get("technical_constraints", {})
    max_timeout = constraints.get("max_bash_execution_time", 300)
    forbidden_commands = constraints.get("forbidden_commands", [])

    # Initial keyword-based risk detection
    keyword_risk, keyword_factors = detect_keyword_risk(original_query)

    # Build the prompt
    # Build the prompt
    # Note: SYSTEM_PROMPT is already an f-string template or partial string
    # We need to be careful with double formatting.
    # The simplest way is to append dynamic parts.

    # Build the prompt
    dynamic_context = f"""
# DYNAMIC CONTEXT

### Workspace Context:
{json.dumps(workspace_context, indent=2)}

### Contraintes Techniques:
- Environnement: Docker sandbox isolé
- Timeout maximum par commande: {max_timeout}s
- Répertoire de travail: /workspace
- Commandes interdites: {", ".join(forbidden_commands)}
"""
    prompt = SYSTEM_PROMPT + dynamic_context

    user_message = f"""Analyse et optimise cette requête utilisateur:

REQUÊTE ORIGINALE:
{original_query}

Réponds avec un JSON valide uniquement."""

    try:
        # Call LLM with structured output
        llm = create_llm(temperature=0.3)  # Lower temperature for precision
        structured_llm = llm.with_structured_output(EnhancerOutput)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]

        # invoke now returns the Pydantic object directly
        parsed = structured_llm.invoke(messages)

        if parsed is None:
            raise ValueError("LLM returned None for structured output")

        logger.debug(f"Enhancer LLM structured response received")

        # Parse response (redundant now, but keeping var name 'parsed')
        # parsed = parse_enhancer_response(response_text) # Removed

        if parsed:
            # Merge keyword-detected risks with LLM-detected risks
            all_risk_factors = keyword_factors + [
                rf.model_dump() for rf in parsed.risk_assessment.risk_factors
            ]

            # Determine final risk level (use the higher of keyword vs LLM detection)
            level_priority = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            final_risk = parsed.risk_assessment.global_level
            if level_priority.get(keyword_risk, 0) > level_priority.get(final_risk, 0):
                final_risk = keyword_risk

            # Determine HITL mode
            current_hitl_mode = state.get("hitl_mode", "moderate")
            final_hitl_mode = determine_hitl_mode(final_risk, current_hitl_mode)

            # Build state updates
            return {
                "enhanced_query": parsed.enhanced_query,
                "detected_intent": parsed.analysis.detected_intent,
                "confidence_score": parsed.analysis.confidence_score,
                "intent_reasoning": parsed.analysis.reasoning,
                "workspace_context": workspace_context,
                "global_risk_level": final_risk,
                "risk_factors": all_risk_factors,
                "hitl_mode": final_hitl_mode,
                "current_breakpoint": "enhanced_prompt_validation",
                "awaiting_human_input": True,
                "messages": [
                    {
                        "role": "assistant",
                        "content": f"[PROMPT ENHANCER]\n"
                        f"Intent: {parsed.analysis.detected_intent} "
                        f"(confidence: {parsed.analysis.confidence_score:.0%})\n"
                        f"Risk Level: {final_risk}\n"
                        f"HITL Mode: {final_hitl_mode}",
                    }
                ],
            }
        else:
            # Parsing failed - use conservative defaults
            logger.warning("Failed to parse enhancer response, using defaults")

            return {
                "enhanced_query": original_query,  # Use original as fallback
                "detected_intent": "mixed_workflow",
                "confidence_score": 0.5,
                "intent_reasoning": "Parsing failed, using conservative defaults",
                "workspace_context": workspace_context,
                "global_risk_level": (
                    keyword_risk if keyword_risk != "low" else "medium"
                ),
                "risk_factors": keyword_factors,
                "hitl_mode": "moderate",
                "current_breakpoint": "enhanced_prompt_validation",
                "awaiting_human_input": True,
                "messages": [
                    {
                        "role": "system",
                        "content": "[PROMPT ENHANCER] Parsing failed, using conservative defaults",
                    }
                ],
            }

    except Exception as e:
        logger.error(f"Prompt enhancer error: {e}")

        # Error fallback - be conservative
        return {
            "enhanced_query": original_query,
            "detected_intent": "mixed_workflow",
            "confidence_score": 0.3,
            "intent_reasoning": f"Error during analysis: {str(e)}",
            "workspace_context": workspace_context,
            "global_risk_level": "medium",
            "risk_factors": keyword_factors
            + [{"type": "analysis_error", "severity": 5, "description": str(e)}],
            "hitl_mode": "moderate",
            "current_breakpoint": "enhanced_prompt_validation",
            "awaiting_human_input": True,
            "error_log": state.get("error_log", [])
            + [{"stage": "prompt_enhancer", "error": str(e)}],
            "messages": [
                {
                    "role": "system",
                    "content": f"[PROMPT ENHANCER ERROR] {str(e)}",
                }
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from agent_state import create_initial_state

    # Test with various queries
    test_queries = [
        "List the files in /workspace",
        "Create a Python script that prints Hello World",
        "Delete all log files older than 30 days",
        "Research the latest AI trends and create a summary",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("=" * 60)

        state = create_initial_state(query)
        result = prompt_enhancer_node(state)

        print(f"Intent: {result.get('detected_intent')}")
        print(f"Confidence: {result.get('confidence_score')}")
        print(f"Risk Level: {result.get('global_risk_level')}")
        print(f"HITL Mode: {result.get('hitl_mode')}")
        print(f"Enhanced Query: {result.get('enhanced_query', '')[:200]}...")
