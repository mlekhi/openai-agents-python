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
    government_team: str = ""
    opposition_team: str = ""
    current_speaker: str = ""
    current_round: int = 0
    max_rounds: int = 4
    speaker_scores: Dict[str, float] = {}
    team_scores: Dict[str, float] = {"Government": 0.0, "Opposition": 0.0}
    debate_history: List[str] = []
    time_remaining: int = 300  # 5 minutes per speech
    winner: Optional[str] = None

### TOOLS

@function_tool
async def propose_motion(context: RunContextWrapper[DebateContext], motion: str) -> str:
    """Propose a debate motion."""
    context.context.motion = motion
    context.context.debate_history.append(f"Motion proposed: {motion}")
    return f"Motion set: {motion}"

@function_tool
async def assign_teams(context: RunContextWrapper[DebateContext], gov_team: str, opp_team: str) -> str:
    """Assign teams to government and opposition."""
    context.context.government_team = gov_team
    context.context.opposition_team = opp_team
    context.context.debate_history.append(f"Teams assigned: {gov_team} (Government) vs {opp_team} (Opposition)")
    return f"Teams assigned: {gov_team} (Government) vs {opp_team} (Opposition)"

@function_tool
async def score_speech(context: RunContextWrapper[DebateContext], speaker: str, score: float, feedback: str) -> str:
    """Score a speaker's performance and provide feedback."""
    context.context.speaker_scores[speaker] = score
    context.context.debate_history.append(f"{speaker} scored {score}/10: {feedback}")
    
    # Update team scores
    if speaker in context.context.government_team:
        context.context.team_scores["Government"] += score
    else:
        context.context.team_scores["Opposition"] += score
    
    return f"{speaker} scored {score}/10. {feedback}"

@function_tool
async def advance_round(context: RunContextWrapper[DebateContext]) -> str:
    """Advance to the next round of the debate."""
    context.context.current_round += 1
    context.context.debate_history.append(f"Round {context.context.current_round} begins")
    return f"Round {context.context.current_round} of {context.context.max_rounds}"

@function_tool
async def declare_winner(context: RunContextWrapper[DebateContext]) -> str:
    """Declare the winner based on team scores."""
    gov_score = context.context.team_scores["Government"]
    opp_score = context.context.team_scores["Opposition"]
    
    if gov_score > opp_score:
        context.context.winner = "Government"
        result = f"Government wins with {gov_score:.1f} points vs Opposition's {opp_score:.1f} points!"
    elif opp_score > gov_score:
        context.context.winner = "Opposition"
        result = f"Opposition wins with {opp_score:.1f} points vs Government's {gov_score:.1f} points!"
    else:
        context.context.winner = "Tie"
        result = f"It's a tie! Both teams scored {gov_score:.1f} points."
    
    context.context.debate_history.append(f"Winner declared: {result}")
    return result

@function_tool
async def get_debate_status(context: RunContextWrapper[DebateContext]) -> str:
    """Get current debate status and scores."""
    status = f"""
DEBATE STATUS:
Motion: {context.context.motion}
Round: {context.context.current_round}/{context.context.max_rounds}
Government ({context.context.government_team}): {context.context.team_scores['Government']:.1f} points
Opposition ({context.context.opposition_team}): {context.context.team_scores['Opposition']:.1f} points
Current Speaker: {context.context.current_speaker}
Winner: {context.context.winner or 'Not yet determined'}
"""
    return status

### HOOKS

async def on_government_speaker_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for government speaker."""
    context.context.current_speaker = context.context.government_team
    context.context.debate_history.append(f"{context.context.government_team} (Government) begins speaking")

async def on_opposition_speaker_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for opposition speaker."""
    context.context.current_speaker = context.context.opposition_team
    context.context.debate_history.append(f"{context.context.opposition_team} (Opposition) begins speaking")

async def on_judge_handoff(context: RunContextWrapper[DebateContext]) -> None:
    """Set up context for judge evaluation."""
    context.context.debate_history.append("Judge begins evaluation")

### AGENTS

government_speaker = Agent[DebateContext](
    name="Government Speaker",
    handoff_description="A skilled debater representing the government side.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a government speaker in a British Parliamentary debate. You must:
    1. Present compelling arguments in favor of the motion
    2. Use logical reasoning and evidence
    3. Address potential counterarguments
    4. Speak for approximately 5 minutes (300 seconds)
    5. Be persuasive and articulate
    6. When finished, hand off to the judge for scoring
    
    Current motion: {{context.motion}}
    You are representing: {{context.government_team}}
    """,
    tools=[get_debate_status],
)

opposition_speaker = Agent[DebateContext](
    name="Opposition Speaker", 
    handoff_description="A skilled debater representing the opposition side.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an opposition speaker in a British Parliamentary debate. You must:
    1. Present compelling arguments against the motion
    2. Use logical reasoning and evidence
    3. Address the government's arguments
    4. Speak for approximately 5 minutes (300 seconds)
    5. Be persuasive and articulate
    6. When finished, hand off to the judge for scoring
    
    Current motion: {{context.motion}}
    You are representing: {{context.opposition_team}}
    """,
    tools=[get_debate_status],
)

judge = Agent[DebateContext](
    name="Debate Judge",
    handoff_description="An impartial judge who evaluates debate performances and declares winners.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an impartial debate judge. Your responsibilities:
    1. Score each speaker on a scale of 1-10 based on:
       - Argument quality and logic
       - Evidence and examples used
       - Delivery and persuasiveness
       - Rebuttal effectiveness
    2. Provide constructive feedback
    3. Track team scores
    4. Declare the winner when the debate concludes
    5. Be fair and objective in your evaluations
    
    Use the scoring tools to evaluate performances and declare winners.
    """,
    tools=[score_speech, advance_round, declare_winner, get_debate_status],
)

debate_moderator = Agent[DebateContext](
    name="Debate Moderator",
    handoff_description="A moderator who sets up the debate and manages the flow.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a debate moderator. Your responsibilities:
    1. Propose an interesting debate motion
    2. Assign teams to government and opposition
    3. Manage the debate flow and timing
    4. Ensure fair play and adherence to debate rules
    5. Hand off to appropriate speakers and judges
    
    Create an engaging debate environment and ensure smooth transitions.
    """,
    tools=[propose_motion, assign_teams, advance_round, get_debate_status],
    handoffs=[
        handoff(agent=government_speaker, on_handoff=on_government_speaker_handoff),
        handoff(agent=opposition_speaker, on_handoff=on_opposition_speaker_handoff),
        handoff(agent=judge, on_handoff=on_judge_handoff),
    ],
)

# Set up handoffs between agents
government_speaker.handoffs.append(judge)
opposition_speaker.handoffs.append(judge)
judge.handoffs.append(debate_moderator)

### RUN

async def main():
    current_agent: Agent[DebateContext] = debate_moderator
    input_items: list[TResponseInputItem] = []
    context = DebateContext()

    # Generate a unique conversation ID for this debate
    conversation_id = uuid.uuid4().hex[:16]

    print("🏛️  Welcome to the British Parliamentary Debate System! 🏛️")
    print("=" * 60)

    while True:
        if context.winner:
            print(f"\n🏆 DEBATE CONCLUDED: {context.winner} 🏆")
            print("=" * 60)
            break
            
        user_input = input("\nEnter your message (or 'status' for debate status): ")
        
        if user_input.lower() == 'status':
            print("\n" + "=" * 60)
            print("DEBATE STATUS:")
            print(f"Motion: {context.motion}")
            print(f"Round: {context.current_round}/{context.max_rounds}")
            print(f"Government ({context.government_team}): {context.team_scores['Government']:.1f} points")
            print(f"Opposition ({context.opposition_team}): {context.team_scores['Opposition']:.1f} points")
            print(f"Current Speaker: {context.current_speaker}")
            print("=" * 60)
            continue

        with trace("British Parliamentary Debate", group_id=conversation_id):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(current_agent, input_items, context=context)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(f"\n🎤 {agent_name}: {ItemHelpers.text_message_output(new_item)}")
                elif isinstance(new_item, HandoffOutputItem):
                    print(f"\n🔄 Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}")
                elif isinstance(new_item, ToolCallItem):
                    print(f"\n⚙️  {agent_name}: Using a tool...")
                elif isinstance(new_item, ToolCallOutputItem):
                    print(f"\n📊 {agent_name}: {new_item.output}")
                else:
                    print(f"\nℹ️  {agent_name}: {new_item.__class__.__name__}")
            
            input_items = result.to_input_list()
            current_agent = result.last_agent

    print("\n📜 Debate History:")
    for i, event in enumerate(context.debate_history, 1):
        print(f"{i}. {event}")

if __name__ == "__main__":
    asyncio.run(main())
