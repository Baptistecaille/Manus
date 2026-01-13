import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml

logger = logging.getLogger(__name__)


class Skill:
    def __init__(self, name: str, path: Path, description: str, instructions: str):
        self.name = name
        self.path = path
        self.description = description
        self.instructions = instructions

    def __repr__(self):
        return f"<Skill {self.name}>"


class SkillManager:
    """
    Manages loading and retrieval of Anthropic-style skills.

    Skills are expected to be in directories under `skills/`.
    Each skill directory must contain a `SKILL.md` file with YAML frontmatter.
    """

    def __init__(self, skills_dir: str = "skills"):
        # Resolve absolute path relative to New_manus root if possible, or use as is
        self.root_dir = Path(os.getcwd())
        self.skills_dir = self.root_dir / skills_dir
        self.skills: Dict[str, Skill] = {}
        self._load_skills()

    def _load_skills(self):
        """Scans the skills directory and loads all valid skills."""
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return

        for item in self.skills_dir.iterdir():
            if item.is_dir():
                skill_file = item / "SKILL.md"
                if skill_file.exists():
                    self._load_single_skill(item.name, skill_file)

    def _load_single_skill(self, name: str, file_path: Path):
        """Parses a single SKILL.md file."""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Simple frontmatter parsing
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_yaml = parts[1]
                instructions = parts[2].strip()

                metadata = yaml.safe_load(frontmatter_yaml)
                description = metadata.get("description", "No description provided.")

                # Prepend the absolute path information to instructions
                # This helps the agent know where the scripts are located
                enriched_instructions = (
                    f"# Skill context\n"
                    f"Skill directory: {file_path.parent}\n"
                    f"Use this path to reference scripts and resources.\n\n"
                    f"{instructions}"
                )

                self.skills[name] = Skill(
                    name=name,
                    path=file_path.parent,
                    description=description,
                    instructions=enriched_instructions,
                )
                logger.info(f"Loaded skill: {name}")
            else:
                logger.warning(f"Invalid SKILL.md format in {name}")

        except Exception as e:
            logger.error(f"Failed to load skill {name}: {e}")

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def get_all_skills_instructions(self) -> str:
        """Returns a concatenated string of all loaded skills' instructions."""
        if not self.skills:
            return ""

        output = ["# Available Specialized Skills\n"]
        for name, skill in self.skills.items():
            output.append(f"## Skill: {name.upper()}")
            output.append(skill.instructions)
            output.append("\n" + "=" * 40 + "\n")

        return "\n".join(output)
