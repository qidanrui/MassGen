# MassGen Case Study: The IMO 2025 Winner

This case study showcases how MassGen (v0.0.1alpha) orchestrated a collaboration between two leading AI models to produce a nuanced, accurate, and well-sourced answer to a complex question.

**Prompt:** `Which AI won IMO 2025?`

**Agents:**
*   Agent 1: `gemini-2.5-flash`
*   Agent 2: `gpt-4.1`

## The Collaborative Process

### Initial Answers

Initially, the two agents provided different answers:

*   **Agent 2 (gpt-4.1)** at first stated that no AI had officially participated in or won the IMO, as it is traditionally a human-only competition.
*   **Agent 1 (gemini-2.5-flash)** correctly stated that both Google DeepMind and OpenAI had achieved gold-level scores, but did not initially differentiate between official and unofficial participation.

### Intelligence Sharing and Refinement

Through MassGen's real-time intelligence sharing, the agents were able to see each other's responses. This led to a rapid refinement of their answers:

1.  **Agent 2**, seeing Agent 1's more current information, updated its answer to reflect the recent news. It went a step further by finding sources and clarifying the crucial distinction between Google's *official* participation and OpenAI's *unofficial* one.
2.  **Agent 1**, while initially correct, was less detailed. It updated its answer to be more comprehensive, but still lacked the clear distinction and sourcing of Agent 2's final answer.

### The Vote: Reaching Consensus

The final phase of the process was the vote. Here's how it played out:

*   **Agent 2** voted for its own answer, citing its superior sourcing, clarity on the official vs. unofficial status, and inclusion of references.
*   **Agent 1** voted for **Agent 2's** answer, acknowledging that it was more accurate and nuanced due to the distinction it made between the two AI's participation.

This resulted in a unanimous consensus, with both agents agreeing that Agent 2's answer was the best.

## The Final Answer

The final, consensus-driven answer was a comprehensive and well-structured summary of the situation, correctly identifying Google DeepMind's Gemini Deep Think as the official winner, while also acknowledging OpenAI's unofficial achievement and the superior performance of the top human contestants. The inclusion of sources and a clear summary table made the final answer highly valuable.

## Conclusion

This case study demonstrates the power of MassGen's collaborative approach. By enabling agents to share information and refine their work in real-time, the system was able to produce a final answer that was more accurate, detailed, and reliable than what either agent could have produced on its own. The consensus-driven process ensured that the best answer was chosen, resulting in a high-quality output for the user.

---

# MassGen Case Study: Synthesis from Diverse Perspectives

This case study demonstrates how MassGen (v0.0.1alpha) handles subjective prompts by tasking a representative agent to synthesize a final, comprehensive answer from multiple, diverse viewpoints.

**Command:**
```bash
python cli.py --config examples/fast_config.yaml "find big AI news this week"
```

**Prompt:** `find big AI news this week`

**Agents:**
*   Agent 1: `gpt-4.1` (Designated Representative Agent)
*   Agent 2: `gemini-2.5-flash`
*   Agent 3: `grok-3-mini`

## The Collaborative Process

### Divergent Interpretations

Given the subjective nature of the prompt, each agent interpreted "big news" differently, leading to three distinct and valuable initial answers:

*   **Agent 1 (gpt-4.1)** focused on high-impact, headline-grabbing news like major policy changes and corporate launches.
*   **Agent 2 (gemini-2.5-flash)** provided a highly detailed, categorized list that included more technical and research-oriented updates.
*   **Agent 3 (grok-3-mini)** prioritized news with clear sources and attempted to provide a broad, balanced summary.

### The Vote: A Tie Between Strong Opinions

Each agent, confident in its own unique and valid interpretation, voted for its own answer. This resulted in a 1-1-1 tie. This outcome highlights the different strengths and priorities of each model, and it provides the system with three different lenses through which to view the week's AI news.

## The Final Answer: Synthesis by the Representative Agent

Instead of simply defaulting to one answer, MassGen's orchestrator designated a `representative_agent` (Agent 1) to perform a final, crucial step: **synthesis**.

Agent 1 was tasked with reviewing its own answer, the answers from Agents 2 and 3, and the voting results. It then created a *new, final answer* that integrated the best elements from all participants. The final output explicitly states it is "**integrating all agents’ highlights**" and combines the policy news from Agent 1, the research breakthroughs from Agent 2, and the sourced consumer news from Agent 3 into a single, cohesive, and comprehensive summary.

## Conclusion

This case study demonstrates a sophisticated feature of MassGen. When a simple consensus isn't possible, the system doesn't fail; it intelligently leverages the diverse outputs to create a synthesized result that is more complete and well-rounded than any single agent's initial response. This makes it exceptionally powerful for exploring complex, subjective topics where multiple viewpoints are not just valid, but essential for a full understanding.

---

# MassGen Case Study: Collaborative Creative Writing

This case study demonstrates MassGen's (v0.0.1) ability to facilitate collaborative creative writing, leading to a consensus on the most compelling narrative.

**Command:**
```bash
python cli.py --config examples/fast_config.yaml "Write a short story about a robot who discovers music."
```

**Prompt:** `Write a short story about a robot who discovers music.`

**Agents:**
*   Agent 1: `gpt-4o`
*   Agent 2: `gemini-2.5-flash`
*   Agent 3: `grok-3-mini`

## The Collaborative Process

### Initial Creative Outputs

Each agent generated a unique short story based on the prompt:

*   **Agent 1 (gpt-4o)** crafted a story about a robot named Evo in Neo-Tokyo who stumbles upon a hidden record shop and discovers music through an old phonograph and a human, Mr. Tanaka. The story emphasizes emotional growth and connection.
*   **Agent 2 (gemini-2.5-flash)** created a narrative about Unit 734, or "Seven," a logic-driven robot that detects an unusual sonic anomaly in a derelict dwelling, leading to the discovery of music and a profound, inexplicable feeling.
*   **Agent 3 (grok-3-mini)** wrote about Echo, a maintenance robot in Neo-City, who encounters a street musician playing a violin in a forgotten alley. Echo learns to improvise and finds choice and creation through music.

### The Vote: Converging on the Best Narrative

This session showcased a clear convergence towards a single best answer, even in a creative task:

1.  **Agent 2** initially voted for its own story, believing it to be the most compelling.
2.  **Agent 1** voted for its own story, citing its completeness, emotional depth, and well-structured narrative.
3.  However, upon reviewing all submissions, **Agent 2 changed its vote to Agent 1's story**, acknowledging its superior structure, emotional resonance, and compelling narrative arc.
4.  **Agent 3** also voted for **Agent 1's story**, praising its completeness, vivid details, human-robot interaction, and overall polish.

This resulted in a strong consensus for Agent 1's story (3 out of 3 votes).

## The Final Answer

Agent 1's story, "Evo's Discovery," was chosen as the final output due to the unanimous consensus. It is a well-developed narrative that effectively addresses the prompt, demonstrating the capacity of MassGen to identify and select high-quality creative content through agent collaboration.

## Conclusion

This case study highlights MassGen's effectiveness in creative tasks. Even with subjective outputs, the multi-agent system can identify and converge on a preferred solution, leveraging the collective judgment of the agents to select the most compelling and well-executed creative piece. This demonstrates MassGen's potential beyond analytical tasks, extending to areas requiring nuanced qualitative assessment.