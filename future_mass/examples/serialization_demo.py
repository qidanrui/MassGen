#!/usr/bin/env python3
"""
Demo showing AgentConfig serialization capabilities.
"""

import json
import sys
from pathlib import Path

# Add the parent directory to the path so we can import mass
sys.path.insert(0, str(Path(__file__).parent.parent))

from mass.agent_config import AgentConfig
from mass.message_templates import MessageTemplates


def demonstrate_serialization():
    """Demonstrate AgentConfig serialization and deserialization."""
    print("💾 AGENT CONFIG SERIALIZATION DEMO")
    print("=" * 60)
    
    # Test 1: Basic config (fully serializable)
    print("\n1. 📦 Basic Configuration")
    basic_config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 4000
        },
        provider_tools=["web_search", "code_interpreter"],
        user_tools=[
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "Query the company database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "SQL query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    )
    
    # Test serialization
    basic_dict = basic_config.to_dict()
    basic_json = json.dumps(basic_dict, indent=2)
    
    print(f"✅ Serialized size: {len(basic_json)} characters")
    print(f"✅ Contains: {len(basic_config.user_tools)} user tools, {len(basic_config.provider_tools)} provider tools")
    
    # Test round-trip
    recovered_basic = AgentConfig.from_dict(json.loads(basic_json))
    print(f"✅ Round-trip successful: {basic_config.backend_params == recovered_basic.backend_params}")
    
    # Test 2: Config with string message templates
    print("\n2. 📝 Configuration with String Templates")
    string_templates = MessageTemplates(
        system_agent_intro="You are {agent_id}, a specialized AI assistant",
        final_answer_instructions="Present your final answer in a clear, structured format"
    )
    
    template_config = AgentConfig(
        backend_params={"model": "claude-3-sonnet-20240229"},
        provider_tools=None,  # All supported tools
        message_templates=string_templates
    )
    
    template_dict = template_config.to_dict()
    template_json = json.dumps(template_dict, indent=2)
    
    print(f"✅ Serialized with templates: {len(template_json)} characters")
    print(f"✅ Template overrides: {len(template_dict.get('message_templates', {}))}")
    
    # Test round-trip
    recovered_template = AgentConfig.from_dict(json.loads(template_json))
    print(f"✅ Templates recovered: {recovered_template.message_templates is not None}")
    
    # Test 3: Config with callable templates (handled gracefully)
    print("\n3. 🚫 Configuration with Callable Templates")
    def custom_intro(agent_id, mode):
        return f"Custom callable intro for {agent_id} in {mode} mode"
    
    callable_templates = MessageTemplates(
        system_agent_intro=custom_intro,  # This is callable
        final_answer_instructions="Present your final answer clearly"
    )
    
    callable_config = AgentConfig(
        backend_params={"model": "grok-beta"},
        message_templates=callable_templates
    )
    
    callable_dict = callable_config.to_dict()
    callable_json = json.dumps(callable_dict, indent=2)
    
    print(f"✅ Serialized with callables: {len(callable_json)} characters")
    print(f"✅ Callable templates handled: {callable_dict.get('message_templates')}")
    
    # Test 4: Save and load from file
    print("\n4. 💾 Save/Load from File")
    
    # Save to file
    config_file = Path("agent_config.json")
    with open(config_file, 'w') as f:
        json.dump(basic_dict, f, indent=2)
    
    print(f"✅ Saved to {config_file}")
    
    # Load from file
    with open(config_file, 'r') as f:
        loaded_dict = json.load(f)
    
    loaded_config = AgentConfig.from_dict(loaded_dict)
    print(f"✅ Loaded from {config_file}")
    print(f"✅ File round-trip successful: {basic_config.backend_params == loaded_config.backend_params}")
    
    # Clean up
    config_file.unlink()
    
    # Test 5: Configuration comparison
    print("\n5. ⚖️  Configuration Comparison")
    
    configs = {
        "GPT-4o Mini": AgentConfig.create_gpt_config(),
        "Claude Sonnet": AgentConfig.create_claude_config(),
        "Grok Beta": AgentConfig.create_grok_config()
    }
    
    for name, config in configs.items():
        config_dict = config.to_dict()
        config_json = json.dumps(config_dict)
        
        print(f"  {name}:")
        print(f"    Model: {config_dict['backend_params'].get('model')}")
        print(f"    Provider tools: {config_dict['provider_tools']}")
        print(f"    JSON size: {len(config_json)} characters")
    
    print("\n" + "=" * 60)
    print("✅ SERIALIZATION SUMMARY:")
    print("=" * 60)
    print("🔹 Fully Serializable:")
    print("  - backend_params (all primitive types)")
    print("  - provider_tools (list of strings or None)")
    print("  - user_tools (list of function schemas)")
    print("  - message_templates (if string-only)")
    
    print("\n🔹 Handled Gracefully:")
    print("  - Callable message_templates (marked as non-serializable)")
    print("  - Complex objects (converted to placeholders)")
    
    print("\n🔹 Round-trip Support:")
    print("  - to_dict() → JSON → from_dict()")
    print("  - File save/load")
    print("  - Configuration comparison")
    
    print("\n🎯 Use Cases:")
    print("  - Save agent configurations to files")
    print("  - Send configs over APIs")
    print("  - Store in databases")
    print("  - Configuration management")


if __name__ == "__main__":
    demonstrate_serialization()