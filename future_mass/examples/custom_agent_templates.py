#!/usr/bin/env python3
"""
Example: Custom Agent Templates
Shows how to create specialized agents with custom message templates.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import mass
sys.path.insert(0, str(Path(__file__).parent.parent))

from mass import MassAgent, AgentConfig
from mass.message_templates import MessageTemplates

def create_medical_specialist():
    """Create a medical specialist agent with custom templates."""
    
    custom_templates = MessageTemplates(
        system_agent_intro="You are {agent_id}, a medical specialist AI with expertise in diagnosis and treatment.",
        
        initial_tool_instructions="""Focus on medical accuracy and evidence-based analysis:
1. Review medical literature and current evidence
2. Analyze symptoms and conditions systematically  
3. Call update_summary() with your medical insights
4. Collaborate with other medical specialists
5. Vote for the most qualified specialist for final diagnosis""",
        
        initial_collaboration_guidance="""You are collaborating with other medical specialists.
- Prioritize patient safety and medical accuracy above all
- Share relevant medical expertise and evidence
- Challenge diagnoses constructively with medical reasoning
- Build consensus on the best treatment approach
- Consider differential diagnosis and rule out serious conditions"""
    )
    
    config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",
            "temperature": 0.2,  # Lower temperature for medical accuracy
            "max_tokens": 6000
        },
        provider_tools=["web_search"],  # For medical literature
        message_templates=custom_templates
    )
    
    return MassAgent("medical_specialist", config)

def create_legal_advisor():
    """Create a legal advisor agent with custom templates."""
    
    # Using callable overrides for more complex logic
    def legal_intro(agent_id, mode):
        if mode == "collaboration":
            return f"You are {agent_id}, a legal advisor specializing in contract law and compliance."
        else:
            return f"You are {agent_id}, presenting legal analysis as the selected counsel."
    
    custom_templates = MessageTemplates(
        system_agent_intro=legal_intro,
        
        initial_tool_instructions="""Approach legal analysis systematically:
1. Identify relevant legal principles and precedents
2. Analyze risks and compliance requirements
3. Call update_summary() with your legal analysis
4. Collaborate with other legal experts
5. Vote for the most qualified legal counsel for final advice""",
        
        initial_collaboration_guidance="""You are working with other legal professionals.
- Ensure legal accuracy and cite relevant authorities
- Consider jurisdictional differences and regulations
- Provide risk assessment and mitigation strategies
- Build consensus on legally sound recommendations
- Flag any compliance or ethical concerns"""
    )
    
    config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",
            "temperature": 0.1,  # Very low for legal precision
            "max_tokens": 8000
        },
        provider_tools=["web_search"],
        message_templates=custom_templates
    )
    
    return MassAgent("legal_advisor", config)

def create_creative_writer():
    """Create a creative writer agent with custom templates."""
    
    custom_templates = MessageTemplates(
        system_agent_intro="You are {agent_id}, a creative writer specializing in storytelling and narrative development.",
        
        initial_tool_instructions="""Approach creative writing with imagination and craft:
1. Develop compelling characters and narratives
2. Explore themes and emotional resonance
3. Call update_summary() with your creative insights
4. Collaborate with other writers and editors
5. Vote for the most creative contributor for final presentation""",
        
        initial_collaboration_guidance="""You are collaborating with other creative professionals.
- Share innovative ideas and creative approaches
- Build on others' creative concepts constructively
- Offer diverse perspectives on storytelling
- Focus on emotional impact and reader engagement
- Balance creativity with narrative coherence"""
    )
    
    config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",
            "temperature": 0.8,  # High temperature for creativity
            "max_tokens": 5000
        },
        provider_tools=["web_search"],
        message_templates=custom_templates
    )
    
    return MassAgent("creative_writer", config)

async def demonstrate_custom_agents():
    """Demonstrate different custom agent configurations."""
    
    print("🎭 CUSTOM AGENT TEMPLATES DEMONSTRATION")
    print("=" * 60)
    
    # Create specialized agents
    medical_agent = create_medical_specialist()
    legal_agent = create_legal_advisor()
    creative_agent = create_creative_writer()
    
    agents = [
        ("Medical Specialist", medical_agent, "Analyze patient symptoms: persistent cough, weight loss, fatigue"),
        ("Legal Advisor", legal_agent, "Review software licensing compliance for enterprise deployment"),
        ("Creative Writer", creative_agent, "Develop a story about AI consciousness and human connection")
    ]
    
    for name, agent, task in agents:
        print(f"\n📋 {name} - Task: {task}")
        print("-" * 40)
        
        # Show how the agent's custom templates affect the work instruction
        work_instruction = agent._build_work_instruction(task)
        print("Work Instruction Preview:")
        print(work_instruction[:300] + "..." if len(work_instruction) > 300 else work_instruction)
        
        # Show system context
        system_context = agent._build_system_context()
        print("\nSystem Context Preview:")
        print(system_context[:200] + "..." if len(system_context) > 200 else system_context)
        print()

async def simple_customization_example():
    """Show simple string-based template customization."""
    
    print("\n🎯 SIMPLE CUSTOMIZATION EXAMPLE")
    print("=" * 60)
    
    # Simple string-based overrides
    simple_templates = MessageTemplates(
        initial_collaboration_guidance="""You are part of a focused analytical team.
- Use data-driven reasoning
- Provide quantitative analysis when possible
- Challenge assumptions with evidence
- Focus on actionable insights"""
    )
    
    config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 4000
        },
        message_templates=simple_templates
    )
    
    agent = MassAgent("analytical_agent", config)
    
    work_instruction = agent._build_work_instruction("Analyze market trends for renewable energy")
    print("Customized Work Instruction:")
    print(work_instruction)

if __name__ == "__main__":
    asyncio.run(demonstrate_custom_agents())
    asyncio.run(simple_customization_example())
    
    print("\n✅ CUSTOM TEMPLATES SUMMARY")
    print("=" * 60)
    print("Custom message templates allow you to:")
    print("• Create domain-specific agents (medical, legal, creative)")
    print("• Customize collaboration styles and approaches")
    print("• Use string templates or callable functions")
    print("• Override any template method without subclassing")
    print("• Maintain full tracing and debugging capabilities")