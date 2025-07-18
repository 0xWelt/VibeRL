"""Generate one-page-per-file documentation matching viberl structure."""

from pathlib import Path

import mkdocs_gen_files


root = Path(__file__).parent.parent
src = root / 'viberl'

# Define the file mapping - each .py file gets its own .md file
file_mapping = {
    # Agents
    'api/agents/base.md': 'viberl.agents.base',
    'api/agents/reinforce.md': 'viberl.agents.reinforce',
    'api/agents/dqn.md': 'viberl.agents.dqn',
    'api/agents/ppo.md': 'viberl.agents.ppo',
    # Networks
    'api/networks/base_network.md': 'viberl.networks.base_network',
    'api/networks/policy_network.md': 'viberl.networks.policy_network',
    'api/networks/value_network.md': 'viberl.networks.value_network',
    # Environments
    'api/envs/grid_world/snake_env.md': 'viberl.envs.grid_world.snake_env',
    # Utils
    'api/utils/common.md': 'viberl.utils.common',
    'api/utils/training.md': 'viberl.utils.training',
    'api/utils/mock_env.md': 'viberl.utils.mock_env',
    'api/utils/experiment_manager.md': 'viberl.utils.experiment_manager',
    'api/utils/vector_env.md': 'viberl.utils.vector_env',
    # Standalone modules
    'api/typing.md': 'viberl.typing',
    'api/cli.md': 'viberl.cli',
}


def generate_module_page(doc_path: str, module_path: str) -> None:
    """Generate a single page for a module showing only module-level content."""

    # Create the page content with only module-level documentation
    content = f"""# `{module_path}`

::: {module_path}
    options:
      show_source: true
      show_root_heading: false
      show_root_full_path: true
      show_bases: true
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      merge_init_into_class: true
      docstring_style: google
      docstring_section_style: table
      show_docstring_description: true
      show_docstring_parameters: true
      show_docstring_returns: true
      show_docstring_raises: true
      inherited_members: false
      members_order: source
      show_submodules: false
      show_category_heading: false
      show_symbol_type_heading: true
      show_symbol_type_toc: true

"""

    with mkdocs_gen_files.open(doc_path, 'w') as fd:
        fd.write(content)


# Generate all pages
for doc_path, module_path in file_mapping.items():
    generate_module_page(doc_path, module_path)

# Create API overview
with mkdocs_gen_files.open('api/index.md', 'w') as fd:
    fd.write("""# API Reference

Welcome to the VibeRL API documentation. This section provides file-level documentation matching the exact structure of the source code.

## File Structure

Each `.py` file in the source code has a corresponding `.md` file in this documentation:

```
viberl/
├── agents/
│   ├── base.py → base.md
│   ├── reinforce.py → reinforce.md
│   ├── dqn.py → dqn.md
│   └── ppo.py → ppo.md
├── networks/
│   ├── base_network.py → base_network.md
│   ├── policy_network.py → policy_network.md
│   └── value_network.py → value_network.md
├── envs/
│   └── grid_world/
│       └── snake_env.py → grid_world/snake_env.md
├── utils/
│   ├── common.py → common.md
│   ├── training.py → training.md
│   ├── mock_env.py → mock_env.md
│   ├── experiment_manager.py → experiment_manager.md
│   └── vector_env.py → vector_env.md
├── typing.py → typing.md
└── cli.py → cli.md
```

## Navigation

Each page contains:
1. **Module Overview** - File-level documentation with hyperlinks
2. **Classes & Functions** - Detailed documentation for all classes and functions in the file

Click on any file in the navigation menu to explore its contents.
""")
