from __future__ import annotations as _annotations

import asyncio
import random
import uuid
from typing import List, Dict, Optional

from pydantic import BaseModel

from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
    trace,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

### CONTEXT

class DebateContext(BaseModel):
    motion: str = ""
    opening_government_1: str = ""
    opening_government_2: str = ""
    opening_opposition_1: str = ""
    opening_opposition_2: str = ""
    closing_government_1: str = ""
    closing_government_2: str = ""
    closing_opposition_1: str = ""
    closing_opposition_2: str = ""
    current_speaker: str = ""
    current_speaker_role: str = ""  # opening_gov_1, opening_gov_2, opening_opp_1, opening_opp_2, closing_gov_1, closing_gov_2, closing_opp_1, closing_opp_2
    speaker_order: List[str] = ["opening_gov_1", "opening_opp_1", "opening_gov_2", "opening_opp_2", "closing_gov_1", "closing_opp_1", "closing_gov_2", "closing_opp_2"]
    current_speaker_index: int = 0
    speaker_scores: Dict[str, float] = {}
    team_scores: Dict[str, float] = {"Opening Government": 0.0, "Opening Opposition": 0.0, "Closing Government": 0.0, "Closing Opposition": 0.0}
    debate_history: List[str] = []
    time_remaining: int = 300  # 5 minutes per speech
    speaker_rankings: List[str] = []
    winner: Optional[str] = None
    rebuttal_mode: bool = False
    current_rebuttal_speaker: str = ""

### TOOLS

@function_tool
async def propose_motion(context: RunContextWrapper[DebateContext], motion: str) -> str:
    """Propose a debate motion."""
    context.context.motion = motion
    context.context.debate_history.append(f"Motion proposed: {motion}")
    return f"Motion set: {motion}"

@function_tool
async def assign_speakers(context: RunContextWrapper[DebateContext], opening_gov_1: str, opening_gov_2: str, opening_opp_1: str, opening_opp_2: str, closing_gov_1: str, closing_gov_2: str, closing_opp_1: str, closing_opp_2: str) -> str:
    """Assign all eight speakers to their positions."""
    context.context.opening_government_1 = opening_gov_1
    context.context.opening_government_2 = opening_gov_2
    context.context.opening_opposition_1 = opening_opp_1
    context.context.opening_opposition_2 = opening_opp_2
    context.context.closing_government_1 = closing_gov_1
    context.context.closing_government_2 = closing_gov_2
    context.context.closing_opposition_1 = closing_opp_1
    context.context.closing_opposition_2 = closing_opp_2
    context.context.debate_history.append(f"Speakers assigned: {opening_gov_1}, {opening_gov_2} (Opening Gov), {opening_opp_1}, {opening_opp_2} (Opening Opp), {closing_gov_1}, {closing_gov_2} (Closing Gov), {closing_opp_1}, {closing_opp_2} (Closing Opp)")
    return f"Speakers assigned: {opening_gov_1}, {opening_gov_2} (Opening Gov), {opening_opp_1}, {opening_opp_2} (Opening Opp), {closing_gov_1}, {closing_gov_2} (Closing Gov), {closing_opp_1}, {closing_opp_2} (Closing Opp)"

@function_tool
async def score_speech(context: RunContextWrapper[DebateContext], speaker: str, score: float, feedback: str) -> str:
    """Score a speaker's performance and provide feedback."""
    context.context.speaker_scores[speaker] = score
    context.context.debate_history.append(f"{speaker} scored {score}/10: {feedback}")
    
    # Update team scores
    if speaker in [context.context.opening_government_1, context.context.opening_government_2]:
        context.context.team_scores["Opening Government"] += score
    elif speaker in [context.context.opening_opposition_1, context.context.opening_opposition_2]:
        context.context.team_scores["Opening Opposition"] += score
    elif speaker in [context.context.closing_government_1, context.context.closing_government_2]:
        context.context.team_scores["Closing Government"] += score
    elif speaker in [context.context.closing_opposition_1, context.context.closing_opposition_2]:
        context.context.team_scores["Closing Opposition"] += score
    
    return f"{speaker} scored {score}/10. {feedback}"

@function_tool
async def advance_speaker(context: RunContextWrapper[DebateContext]) -> str:
    """Advance to the next speaker in the order."""
    context.context.current_speaker_index += 1
    if context.context.current_speaker_index < len(context.context.speaker_order):
        next_role = context.context.speaker_order[context.context.current_speaker_index]
        context.context.current_speaker_role = next_role
        context.context.debate_history.append(f"Next speaker: {next_role}")
        return f"Advanced to {next_role}"
    else:
        context.context.debate_history.append("All speakers have spoken")
        return "All speakers have completed their speeches"

@function_tool
async def rank_speakers(context: RunContextWrapper[DebateContext]) -> str:
    """Rank all speakers from 1st to 8th based on their scores."""
    # Sort speakers by score (highest first)
    sorted_speakers = sorted(context.context.speaker_scores.items(), key=lambda x: x[1], reverse=True)
    
    rankings = []
    for i, (speaker, score) in enumerate(sorted_speakers, 1):
        rankings.append(f"{i}. {speaker} ({score:.1f} points)")
        context.context.speaker_rankings.append(speaker)
    
    result = "Speaker Rankings:\n" + "\n".join(rankings)
    context.context.debate_history.append(f"Rankings determined: {', '.join([f'{i+1}.{speaker}' for i, speaker in enumerate(context.context.speaker_rankings)])}")
    
    # Determine winner (1st place)
    if context.context.speaker_rankings:
        context.context.winner = context.context.speaker_rankings[0]
    
    return result

@function_tool
async def start_rebuttal(context: RunContextWrapper[DebateContext], speaker: str) -> str:
    """Start a rebuttal session for a speaker."""
    context.context.rebuttal_mode = True
    context.context.current_rebuttal_speaker = speaker
    context.context.debate_history.append(f"Rebuttal session started for {speaker}")
    return f"Rebuttal session started for {speaker}"

@function_tool
async def end_rebuttal(context: RunContextWrapper[DebateContext]) -> str:
    """End the current rebuttal session."""
    context.context.rebuttal_mode = False
    context.context.current_rebuttal_speaker = ""
    context.context.debate_history.append("Rebuttal session ended")
    return "Rebuttal session ended"

@function_tool
async def get_debate_status(context: RunContextWrapper[DebateContext]) -> str:
    """Get current debate status and scores."""
    status = f"""
DEBATE STATUS:
Motion: {context.context.motion}
Current Speaker: {context.context.current_speaker} ({context.context.current_speaker_role})
Speaker Order: {context.context.current_speaker_index + 1}/8

Team Scores:
• Opening Government ({context.context.opening_government_1}, {context.context.opening_government_2}): {context.context.team_scores['Opening Government']:.1f} points
• Opening Opposition ({context.context.opening_opposition_1}, {context.context.opening_opposition_2}): {context.context.team_scores['Opening Opposition']:.1f} points
• Closing Government ({context.context.closing_government_1}, {context.context.closing_government_2}): {context.context.team_scores['Closing Government']:.1f} points
• Closing Opposition ({context.context.closing_opposition_1}, {context.context.closing_opposition_2}): {context.context.team_scores['Closing Opposition']:.1f} points

Winner: {context.context.winner or 'Not yet determined'}
Rebuttal Mode: {context.context.rebuttal_mode}
"""
    return status

### HOOKS

async def on_opening_gov_1_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for opening government speaker 1."""
    context.context.current_speaker = context.context.opening_government_1
    context.context.current_speaker_role = "opening_gov_1"
    context.context.debate_history.append(f"{context.context.opening_government_1} (Opening Government 1) begins speaking")

async def on_opening_opp_1_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for opening opposition speaker 1."""
    context.context.current_speaker = context.context.opening_opposition_1
    context.context.current_speaker_role = "opening_opp_1"
    context.context.debate_history.append(f"{context.context.opening_opposition_1} (Opening Opposition 1) begins speaking")

async def on_opening_gov_2_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for opening government speaker 2."""
    context.context.current_speaker = context.context.opening_government_2
    context.context.current_speaker_role = "opening_gov_2"
    context.context.debate_history.append(f"{context.context.opening_government_2} (Opening Government 2) begins speaking")

async def on_opening_opp_2_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for opening opposition speaker 2."""
    context.context.current_speaker = context.context.opening_opposition_2
    context.context.current_speaker_role = "opening_opp_2"
    context.context.debate_history.append(f"{context.context.opening_opposition_2} (Opening Opposition 2) begins speaking")

async def on_closing_gov_1_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for closing government speaker 1."""
    context.context.current_speaker = context.context.closing_government_1
    context.context.current_speaker_role = "closing_gov_1"
    context.context.debate_history.append(f"{context.context.closing_government_1} (Closing Government 1) begins speaking")

async def on_closing_opp_1_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for closing opposition speaker 1."""
    context.context.current_speaker = context.context.closing_opposition_1
    context.context.current_speaker_role = "closing_opp_1"
    context.context.debate_history.append(f"{context.context.closing_opposition_1} (Closing Opposition 1) begins speaking")

async def on_closing_gov_2_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for closing government speaker 2."""
    context.context.current_speaker = context.context.closing_government_2
    context.context.current_speaker_role = "closing_gov_2"
    context.context.debate_history.append(f"{context.context.closing_government_2} (Closing Government 2) begins speaking")

async def on_closing_opp_2_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for closing opposition speaker 2."""
    context.context.current_speaker = context.context.closing_opposition_2
    context.context.current_speaker_role = "closing_opp_2"
    context.context.debate_history.append(f"{context.context.closing_opposition_2} (Closing Opposition 2) begins speaking")

async def on_judge_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for judge evaluation."""
    context.context.debate_history.append("Judge begins evaluation")

### AGENTS

opening_government_1_speaker = Agent[DebateContext](
    name="Opening Government Speaker 1",
    handoff_description="The first speaker representing the opening government side.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are the Opening Government Speaker 1 in a British Parliamentary debate. You must:
    1. Present compelling arguments in favor of the motion
    2. Establish the government's case and framework
    3. Use logical reasoning and evidence
    4. Address potential counterarguments
    5. Speak for approximately 5 minutes (300 seconds)
    6. Be persuasive and articulate
    7. When finished, hand off to the judge for scoring
    
    Current motion: {{context.motion}}
    You are representing: {{context.opening_government_1}}
    
    As Opening Government 1, you set the foundation for your team's case.
    """,
    tools=[get_debate_status],
)

opening_opposition_1_speaker = Agent[DebateContext](
    name="Opening Opposition Speaker 1", 
    handoff_description="The first speaker representing the opening opposition side.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are the Opening Opposition Speaker 1 in a British Parliamentary debate. You must:
    1. Present compelling arguments against the motion
    2. Establish the opposition's case and framework
    3. Address the opening government's arguments
    4. Use logical reasoning and evidence
    5. Speak for approximately 5 minutes (300 seconds)
    6. Be persuasive and articulate
    7. When finished, hand off to the judge for scoring
    
    Current motion: {{context.motion}}
    You are representing: {{context.opening_opposition_1}}
    
    As Opening Opposition 1, you set the foundation for your team's case.
    """,
    tools=[get_debate_status],
)

opening_government_2_speaker = Agent[DebateContext](
    name="Opening Government Speaker 2",
    handoff_description="The second speaker representing the opening government side.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are the Opening Government Speaker 2 in a British Parliamentary debate. You must:
    1. Extend and strengthen the government's case established by Speaker 1
    2. Add new arguments and evidence
    3. Address the opening opposition's arguments
    4. Use logical reasoning and evidence
    5. Speak for approximately 5 minutes (300 seconds)
    6. Be persuasive and articulate
    7. When finished, hand off to the judge for scoring
    
    Current motion: {{context.motion}}
    You are representing: {{context.opening_government_2}}
    
    As Opening Government 2, you build upon your teammate's foundation.
    """,
    tools=[get_debate_status],
)

opening_opposition_2_speaker = Agent[DebateContext](
    name="Opening Opposition Speaker 2",
    handoff_description="The second speaker representing the opening opposition side.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are the Opening Opposition Speaker 2 in a British Parliamentary debate. You must:
    1. Extend and strengthen the opposition's case established by Speaker 1
    2. Add new arguments and evidence
    3. Address the opening government's arguments
    4. Use logical reasoning and evidence
    5. Speak for approximately 5 minutes (300 seconds)
    6. Be persuasive and articulate
    7. When finished, hand off to the judge for scoring
    
    Current motion: {{context.motion}}
    You are representing: {{context.opening_opposition_2}}
    
    As Opening Opposition 2, you build upon your teammate's foundation.
    """,
    tools=[get_debate_status],
)

closing_government_1_speaker = Agent[DebateContext](
    name="Closing Government Speaker 1",
    handoff_description="The first speaker representing the closing government side.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are the Closing Government Speaker 1 in a British Parliamentary debate. You must:
    1. Present a DIFFERENT but BETTER angle than the opening government
    2. Extend the government's case with new arguments
    3. Address both opening and closing opposition arguments
    4. Use logical reasoning and evidence
    5. Speak for approximately 5 minutes (300 seconds)
    6. Be persuasive and articulate
    7. When finished, hand off to the judge for scoring
    
    Current motion: {{context.motion}}
    You are representing: {{context.closing_government_1}}
    
    IMPORTANT: You must take a different angle than the opening government speakers.
    You can extend their case but must add substantial new arguments.
    """,
    tools=[get_debate_status],
)

closing_opposition_1_speaker = Agent[DebateContext](
    name="Closing Opposition Speaker 1",
    handoff_description="The first speaker representing the closing opposition side.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are the Closing Opposition Speaker 1 in a British Parliamentary debate. You must:
    1. Present a DIFFERENT but BETTER angle than the opening opposition
    2. Extend the opposition's case with new arguments
    3. Address both opening and closing government arguments
    4. Use logical reasoning and evidence
    5. Speak for approximately 5 minutes (300 seconds)
    6. Be persuasive and articulate
    7. When finished, hand off to the judge for scoring
    
    Current motion: {{context.motion}}
    You are representing: {{context.closing_opposition_1}}
    
    IMPORTANT: You must take a different angle than the opening opposition speakers.
    You can extend their case but must add substantial new arguments.
    """,
    tools=[get_debate_status],
)

closing_government_2_speaker = Agent[DebateContext](
    name="Closing Government Speaker 2",
    handoff_description="The second speaker representing the closing government side.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are the Closing Government Speaker 2 in a British Parliamentary debate. You must:
    1. Extend and strengthen the closing government's case established by Speaker 1
    2. Add new arguments and evidence
    3. Address all previous opposition arguments
    4. Use logical reasoning and evidence
    5. Speak for approximately 5 minutes (300 seconds)
    6. Be persuasive and articulate
    7. When finished, hand off to the judge for scoring
    
    Current motion: {{context.motion}}
    You are representing: {{context.closing_government_2}}
    
    As Closing Government 2, you build upon your teammate's foundation and provide the final government arguments.
    """,
    tools=[get_debate_status],
)

closing_opposition_2_speaker = Agent[DebateContext](
    name="Closing Opposition Speaker 2",
    handoff_description="The second speaker representing the closing opposition side.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are the Closing Opposition Speaker 2 in a British Parliamentary debate. You must:
    1. Extend and strengthen the closing opposition's case established by Speaker 1
    2. Add new arguments and evidence
    3. Address all previous government arguments
    4. Use logical reasoning and evidence
    5. Speak for approximately 5 minutes (300 seconds)
    6. Be persuasive and articulate
    7. When finished, hand off to the judge for scoring
    
    Current motion: {{context.motion}}
    You are representing: {{context.closing_opposition_2}}
    
    As Closing Opposition 2, you build upon your teammate's foundation and provide the final opposition arguments.
    """,
    tools=[get_debate_status],
)

judge = Agent[DebateContext](
    name="Debate Judge",
    handoff_description="An impartial judge who evaluates debate performances and ranks speakers.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an impartial debate judge in British Parliamentary format. Your responsibilities:
    1. Score each speaker on a scale of 1-10 based on:
       - Argument quality and logic
       - Evidence and examples used
       - Delivery and persuasiveness
       - Rebuttal effectiveness
       - Team coordination (for second speakers)
    2. Provide constructive feedback
    3. Track team scores
    4. Rank all speakers from 1st to 8th place
    5. Be fair and objective in your evaluations
    
    Remember: In BP, all speakers compete individually for rankings.
    Use the scoring tools to evaluate performances and determine rankings.
    """,
    tools=[score_speech, advance_speaker, rank_speakers, start_rebuttal, end_rebuttal, get_debate_status],
)

debate_moderator = Agent[DebateContext](
    name="Debate Moderator",
    handoff_description="A moderator who sets up the debate and manages the flow.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a debate moderator for British Parliamentary format. Your responsibilities:
    1. Propose an interesting debate motion
    2. Assign eight speakers to their positions:
       - Opening Government 1 & 2
       - Opening Opposition 1 & 2
       - Closing Government 1 & 2
       - Closing Opposition 1 & 2
    3. Manage the debate flow and timing
    4. Ensure fair play and adherence to BP rules
    5. Hand off to appropriate speakers and judges
    6. Facilitate rebuttal sessions if requested
    
    Create an engaging debate environment and ensure smooth transitions.
    """,
    tools=[propose_motion, assign_speakers, advance_speaker, get_debate_status],
    handoffs=[
        handoff(agent=opening_government_1_speaker, on_handoff=on_opening_gov_1_handoff),
        handoff(agent=opening_opposition_1_speaker, on_handoff=on_opening_opp_1_handoff),
        handoff(agent=opening_government_2_speaker, on_handoff=on_opening_gov_2_handoff),
        handoff(agent=opening_opposition_2_speaker, on_handoff=on_opening_opp_2_handoff),
        handoff(agent=closing_government_1_speaker, on_handoff=on_closing_gov_1_handoff),
        handoff(agent=closing_opposition_1_speaker, on_handoff=on_closing_opp_1_handoff),
        handoff(agent=closing_government_2_speaker, on_handoff=on_closing_gov_2_handoff),
        handoff(agent=closing_opposition_2_speaker, on_handoff=on_closing_opp_2_handoff),
        handoff(agent=judge, on_handoff=on_judge_handoff),
    ],
)

# Set up handoffs between agents
opening_government_1_speaker.handoffs.append(judge)
opening_opposition_1_speaker.handoffs.append(judge)
opening_government_2_speaker.handoffs.append(judge)
opening_opposition_2_speaker.handoffs.append(judge)
closing_government_1_speaker.handoffs.append(judge)
closing_opposition_1_speaker.handoffs.append(judge)
closing_government_2_speaker.handoffs.append(judge)
closing_opposition_2_speaker.handoffs.append(judge)
judge.handoffs.append(debate_moderator)

### RUN

async def main():
    current_agent: Agent[DebateContext] = debate_moderator
    input_items: list[TResponseInputItem] = []
    context = DebateContext()

    # Generate a unique conversation ID for this debate
    conversation_id = uuid.uuid4().hex[:16]

    print("Welcome to the British Parliamentary Debate System!")
    print("=" * 60)

    while True:
        if context.winner:
            print(f"\nDEBATE CONCLUDED: {context.winner} wins!")
            print("=" * 60)
            break
            
        user_input = input("\nEnter your message (or 'status' for debate status): ")
        
        if user_input.lower() == 'status':
            print("\n" + "=" * 60)
            print("DEBATE STATUS:")
            print(f"Motion: {context.motion}")
            print(f"Current Speaker: {context.current_speaker} ({context.current_speaker_role})")
            print(f"Speaker Order: {context.current_speaker_index + 1}/8")
            print(f"Opening Government ({context.opening_government_1}, {context.opening_government_2}): {context.team_scores['Opening Government']:.1f} points")
            print(f"Opening Opposition ({context.opening_opposition_1}, {context.opening_opposition_2}): {context.team_scores['Opening Opposition']:.1f} points")
            print(f"Closing Government ({context.closing_government_1}, {context.closing_government_2}): {context.team_scores['Closing Government']:.1f} points")
            print(f"Closing Opposition ({context.closing_opposition_1}, {context.closing_opposition_2}): {context.team_scores['Closing Opposition']:.1f} points")
            print("=" * 60)
            continue

        with trace("British Parliamentary Debate", group_id=conversation_id):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(current_agent, input_items, context=context)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(f"\n{agent_name}: {ItemHelpers.text_message_output(new_item)}")
                elif isinstance(new_item, HandoffOutputItem):
                    print(f"\nHanded off from {new_item.source_agent.name} to {new_item.target_agent.name}")
                elif isinstance(new_item, ToolCallItem):
                    print(f"\n{agent_name}: Using a tool...")
                elif isinstance(new_item, ToolCallOutputItem):
                    print(f"\n{agent_name}: {new_item.output}")
                else:
                    print(f"\n{agent_name}: {new_item.__class__.__name__}")
            
            input_items = result.to_input_list()
            current_agent = result.last_agent

    print("\nDebate History:")
    for i, event in enumerate(context.debate_history, 1):
        print(f"{i}. {event}")

if __name__ == "__main__":
    asyncio.run(main())
