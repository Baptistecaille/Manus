"""
CLI Interface - Rich-based terminal UI for HITL interactions.

Provides formatted displays for breakpoints and user input handling.
"""

import sys
import threading
import time
from typing import Literal, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.syntax import Syntax
from rich import box

from agent_state import AgentStateDict
from hitl.breakpoints import BreakpointConfig, BreakpointResult, BreakpointType

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def display_prompt_validation(state: AgentStateDict) -> None:
    """Display the enhanced prompt validation interface."""

    console.print()
    console.rule("[bold blue]🔍 VALIDATION DE LA REQUÊTE OPTIMISÉE[/bold blue]")
    console.print()

    # Original query
    original = state.get("original_query", "")
    console.print(
        Panel(
            original,
            title="[bold]REQUÊTE ORIGINALE[/bold]",
            border_style="dim",
        )
    )

    # Enhanced query
    enhanced = state.get("enhanced_query", "")
    console.print(
        Panel(
            enhanced or "[dim]Pas de modification[/dim]",
            title="[bold green]REQUÊTE OPTIMISÉE[/bold green]",
            border_style="green",
        )
    )

    # Analysis info
    info_table = Table(box=box.SIMPLE, show_header=False)
    info_table.add_column("Field", style="bold")
    info_table.add_column("Value")

    info_table.add_row("Intent", state.get("detected_intent", "unknown"))
    info_table.add_row("Confidence", f"{state.get('confidence_score', 0):.0%}")
    info_table.add_row("Reasoning", state.get("intent_reasoning", "")[:100] + "...")

    console.print(info_table)
    console.print()

    # Risk assessment
    risk_level = state.get("global_risk_level", "low")
    risk_colors = {
        "low": "green",
        "medium": "yellow",
        "high": "red",
        "critical": "bold red",
    }
    risk_emoji = {
        "low": "🟢",
        "medium": "🟡",
        "high": "🔴",
        "critical": "⚫",
    }

    risk_text = Text()
    risk_text.append(f"{risk_emoji.get(risk_level, '❓')} Niveau: ", style="bold")
    risk_text.append(risk_level.upper(), style=risk_colors.get(risk_level, "white"))

    console.print(
        Panel(
            risk_text,
            title="[bold]ANALYSE DE RISQUE[/bold]",
            border_style=risk_colors.get(risk_level, "white"),
        )
    )

    # Show risk factors if any
    risk_factors = state.get("risk_factors", [])
    if risk_factors:
        for factor in risk_factors[:3]:  # Limit to 3
            console.print(f"  • {factor.get('description', '')}")
        if len(risk_factors) > 3:
            console.print(f"  [dim]... et {len(risk_factors) - 3} autres[/dim]")

    console.print()


def display_plan_validation(state: AgentStateDict) -> None:
    """Display the execution plan validation interface."""

    console.print()
    console.rule("[bold blue]📋 VALIDATION DU PLAN D'EXÉCUTION[/bold blue]")
    console.print()

    # Objective
    objective = state.get("enhanced_query", state.get("original_query", ""))[:200]
    console.print(f"[bold]OBJECTIF:[/bold] {objective}")
    console.print()

    # Execution plan
    plan = state.get("execution_plan", {})
    steps = plan.get("steps", [])

    if not steps:
        console.print("[dim]Aucun plan d'exécution défini[/dim]")
        return

    console.print("[bold]ÉTAPES PLANIFIÉES:[/bold]")
    console.print()

    for i, step in enumerate(steps, 1):
        risk_emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🔴",
        }.get(step.get("estimated_risk", "low"), "❓")

        step_table = Table(box=box.ROUNDED, show_header=False, width=60)
        step_table.add_column("Field", width=15)
        step_table.add_column("Value", width=40)

        step_table.add_row("Description", step.get("description", "N/A"))
        step_table.add_row("Executor", step.get("executor", "bash"))
        step_table.add_row(
            "Risque", f"{risk_emoji} {step.get('estimated_risk', 'low').upper()}"
        )

        console.print(
            Panel(
                step_table,
                title=f"[bold]ÉTAPE {i}[/bold]",
                border_style="blue",
            )
        )

    # Summary
    duration = plan.get("total_estimated_duration", 0)
    console.print()
    console.print(f"[bold]DURÉE TOTALE ESTIMÉE:[/bold] {duration} secondes")
    console.print()


def display_bash_validation(command: str, risk_level: str, justification: str) -> None:
    """Display the bash command validation interface."""

    console.print()
    console.rule("[bold yellow]⚠️ VALIDATION DE COMMANDE BASH[/bold yellow]")
    console.print()

    # Command with syntax highlighting
    console.print(
        Panel(
            Syntax(command, "bash", theme="monokai", line_numbers=False),
            title="[bold]COMMANDE À EXÉCUTER[/bold]",
            border_style="yellow",
        )
    )

    # Risk level
    risk_colors = {
        "low": "green",
        "medium": "yellow",
        "high": "red",
        "critical": "bold red",
    }
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "⚫"}

    console.print()
    console.print(f"[bold]ANALYSE DE SÉCURITÉ:[/bold]")
    console.print(
        f"  {risk_emoji.get(risk_level, '❓')} Niveau de risque: [{risk_colors.get(risk_level, 'white')}]{risk_level.upper()}[/{risk_colors.get(risk_level, 'white')}]"
    )
    console.print()

    # Justification
    if justification:
        console.print(
            Panel(
                justification,
                title="[bold]JUSTIFICATION DE L'AGENT[/bold]",
                border_style="dim",
            )
        )

    console.print()


def display_actions(breakpoint_type: BreakpointType) -> None:
    """Display available actions for the current breakpoint."""

    actions_table = Table(box=box.SIMPLE, show_header=False)
    actions_table.add_column("Key", style="bold cyan", width=5)
    actions_table.add_column("Action", width=50)

    actions_table.add_row("[A]", "Approuver et continuer")
    actions_table.add_row("[M]", "Modifier la requête/commande")

    if breakpoint_type == BreakpointType.PLAN_VALIDATION:
        actions_table.add_row("[S]", "Sauter une étape")

    actions_table.add_row("[R]", "Rejeter")
    actions_table.add_row("[I]", "Plus d'informations")
    actions_table.add_row("[Q]", "Quitter")

    console.print(
        Panel(
            actions_table,
            title="[bold]ACTIONS DISPONIBLES[/bold]",
            border_style="cyan",
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT HANDLING
# ═══════════════════════════════════════════════════════════════════════════════


def get_user_decision(
    config: BreakpointConfig,
    timeout_enabled: bool = True,
) -> BreakpointResult:
    """
    Get user decision with optional timeout.

    Args:
        config: Breakpoint configuration with timeout settings.
        timeout_enabled: Whether to enforce timeout.

    Returns:
        BreakpointResult with user's decision.
    """
    start_time = time.time()
    user_input = None
    timed_out = False

    valid_choices = {
        "a": "approve",
        "m": "modify",
        "r": "reject",
        "i": "info",
        "q": "quit",
        "s": "skip",
    }

    if timeout_enabled:
        # Show timeout countdown
        console.print()
        console.print(
            f"[dim]⏱️  Décision requise dans {config.timeout_seconds} secondes[/dim]"
        )
        console.print(
            f"[dim]   (action par défaut: {config.default_action.upper()})[/dim]"
        )
        console.print()

    try:
        while True:
            user_input = Prompt.ask(
                "Votre choix",
                choices=["a", "m", "r", "i", "q", "s", "A", "M", "R", "I", "Q", "S"],
                default="a" if config.default_action == "approve" else "r",
            ).lower()

            if user_input in valid_choices:
                break
            console.print("[red]Choix invalide. Veuillez réessayer.[/red]")

    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Interruption reçue. Quitter...[/yellow]")
        user_input = "q"

    response_time = time.time() - start_time

    action = valid_choices.get(user_input, config.default_action)

    # Handle the "info" action - show more details and ask again
    if action == "info":
        console.print(
            "\n[dim]Informations supplémentaires requises - fonctionnalité à implémenter[/dim]\n"
        )
        return get_user_decision(config, timeout_enabled=False)

    # Handle modification
    modifications = {}
    feedback = ""

    if action == "modify":
        console.print()
        feedback = Prompt.ask("Entrez votre modification ou commentaire")
        modifications["user_feedback"] = feedback

    return BreakpointResult(
        action=action,
        feedback=feedback,
        modifications=modifications,
        response_time_seconds=response_time,
        timed_out=timed_out,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DISPLAY ROUTER
# ═══════════════════════════════════════════════════════════════════════════════


def display_breakpoint(
    breakpoint_type: BreakpointType,
    state: AgentStateDict,
    command: str = "",
    risk_level: str = "low",
    justification: str = "",
) -> None:
    """
    Display the appropriate interface for a breakpoint type.

    Args:
        breakpoint_type: Type of breakpoint.
        state: Current agent state.
        command: Command string (for bash validation).
        risk_level: Risk level.
        justification: Agent's justification (for bash validation).
    """
    if breakpoint_type == BreakpointType.ENHANCED_PROMPT:
        display_prompt_validation(state)
    elif breakpoint_type == BreakpointType.PLAN_VALIDATION:
        display_plan_validation(state)
    elif breakpoint_type == BreakpointType.BASH_COMMAND:
        display_bash_validation(command, risk_level, justification)

    display_actions(breakpoint_type)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO MODE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Demo mode for testing the CLI interface
    console.print("\n[bold]HITL CLI Interface Demo[/bold]\n")

    # Create mock state
    mock_state = {
        "original_query": "Supprime tous les fichiers logs de plus de 30 jours",
        "enhanced_query": "Générer un script Bash sécurisé qui:\n1. Identifie les fichiers *.log dans /var/log\n2. Filtre ceux modifiés il y a >30 jours\n3. Les supprime APRÈS confirmation interactive\n4. Génère un rapport de suppression dans cleanup_report.txt",
        "detected_intent": "file_manipulation",
        "confidence_score": 0.92,
        "intent_reasoning": "La requête demande explicitement la suppression de fichiers basée sur leur âge.",
        "global_risk_level": "medium",
        "risk_factors": [
            {
                "type": "sensitive_keyword",
                "severity": 7,
                "description": "Utilisation de 'rm' détectée",
            },
            {
                "type": "file_modification",
                "severity": 5,
                "description": "Modification du système de fichiers",
            },
        ],
        "execution_plan": {
            "steps": [
                {
                    "description": "Identifier les fichiers logs",
                    "executor": "bash",
                    "estimated_risk": "low",
                },
                {
                    "description": "Générer le script de nettoyage",
                    "executor": "code_generation",
                    "estimated_risk": "low",
                },
                {
                    "description": "Exécuter en mode dry-run",
                    "executor": "bash",
                    "estimated_risk": "medium",
                },
            ],
            "total_estimated_duration": 30,
        },
    }

    # Test prompt validation
    console.print("[bold cyan]--- Test: Prompt Validation ---[/bold cyan]")
    display_breakpoint(BreakpointType.ENHANCED_PROMPT, mock_state)

    # Test plan validation
    console.print("[bold cyan]--- Test: Plan Validation ---[/bold cyan]")
    display_breakpoint(BreakpointType.PLAN_VALIDATION, mock_state)

    # Test bash validation
    console.print("[bold cyan]--- Test: Bash Command Validation ---[/bold cyan]")
    display_breakpoint(
        BreakpointType.BASH_COMMAND,
        mock_state,
        command="find /var/log -name '*.log' -mtime +30 -exec rm {} \\;",
        risk_level="high",
        justification="Cette commande supprime les fichiers logs de plus de 30 jours conformément à la requête utilisateur.",
    )
