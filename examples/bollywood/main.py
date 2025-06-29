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

class BollywoodArgumentContext(BaseModel):
    argument_topic: str = ""
    character_1_name: str = ""
    character_2_name: str = ""
    character_1_role: str = ""
    character_2_role: str = ""
    current_speaker: str = ""
    argument_history: List[Dict] = []
    current_round: int = 0
    dramatic_intensity: int = 5  # 1-10 scale
    user_questions: List[str] = []
    translations_given: List[str] = []

### TOOLS

@function_tool
async def set_argument_scene(context: RunContextWrapper[BollywoodArgumentContext], topic: str, char1_name: str, char1_role: str, char2_name: str, char2_role: str) -> str:
    """Set up the dramatic argument scene with characters and topic."""
    context.context.argument_topic = topic
    context.context.character_1_name = char1_name
    context.context.character_2_name = char2_name
    context.context.character_1_role = char1_role
    context.context.character_2_role = char2_role
    context.context.argument_history.append({
        "type": "scene_setup",
        "topic": topic,
        "char1": f"{char1_name} ({char1_role})",
        "char2": f"{char2_name} ({char2_role})"
    })
    return f"Scene set: {char1_name} ({char1_role}) vs {char2_name} ({char2_role}) arguing about {topic}"

@function_tool
async def speak_hindi_dialogue(context: RunContextWrapper[BollywoodArgumentContext], speaker: str, hindi_dialogue: str, english_translation: str, dramatic_level: int) -> str:
    """Speak a dramatic Hindi dialogue with English translation."""
    context.context.current_speaker = speaker
    context.context.current_round += 1
    context.context.dramatic_intensity = dramatic_level
    
    context.context.argument_history.append({
        "type": "dialogue",
        "speaker": speaker,
        "hindi": hindi_dialogue,
        "english": english_translation,
        "dramatic_level": dramatic_level,
        "round": context.context.current_round
    })
    
    return f"{speaker}: \"{hindi_dialogue}\"\nTranslation: \"{english_translation}\"\nDramatic Level: {dramatic_level}/10"

@function_tool
async def explain_meaning(context: RunContextWrapper[BollywoodArgumentContext], word_or_phrase: str, detailed_explanation: str) -> str:
    """Provide detailed explanation of Hindi words or phrases."""
    context.context.user_questions.append(word_or_phrase)
    context.context.translations_given.append(detailed_explanation)
    return f"Explanation of '{word_or_phrase}':\n{detailed_explanation}"

@function_tool
async def escalate_drama(context: RunContextWrapper[BollywoodArgumentContext], new_intensity: int) -> str:
    """Escalate the dramatic intensity of the argument."""
    context.context.dramatic_intensity = new_intensity
    context.context.argument_history.append({
        "type": "drama_escalation",
        "new_intensity": new_intensity
    })
    return f"Dramatic intensity escalated to {new_intensity}/10"

@function_tool
async def get_argument_status(context: RunContextWrapper[BollywoodArgumentContext]) -> str:
    """Get current argument status and history."""
    status = f"""
ARGUMENT STATUS:
Topic: {context.context.argument_topic}
Characters: {context.context.character_1_name} ({context.context.character_1_role}) vs {context.context.character_2_name} ({context.context.character_2_role})
Current Speaker: {context.context.current_speaker}
Current Round: {context.context.current_round}
Dramatic Intensity: {context.context.dramatic_intensity}/10
User Questions Asked: {len(context.context.user_questions)}
"""
    return status

@function_tool
async def add_bollywood_element(context: RunContextWrapper[BollywoodArgumentContext], element_type: str, description: str) -> str:
    """Add a dramatic Bollywood element to the scene."""
    context.context.argument_history.append({
        "type": "bollywood_element",
        "element": element_type,
        "description": description
    })
    return f"Added {element_type}: {description}"

### HOOKS

async def on_character_1_handoff(context: RunContextWrapper[BollywoodArgumentContext]) -> None:
    """Set up context for character 1."""
    context.context.current_speaker = context.context.character_1_name
    context.context.argument_history.append({"type": "speaker_change", "speaker": context.context.character_1_name})

async def on_character_2_handoff(context: RunContextWrapper[BollywoodArgumentContext]) -> None:
    """Set up context for character 2."""
    context.context.current_speaker = context.context.character_2_name
    context.context.argument_history.append({"type": "speaker_change", "speaker": context.context.character_2_name})

async def on_translator_handoff(context: RunContextWrapper[BollywoodArgumentContext]) -> None:
    """Set up context for translator."""
    context.context.argument_history.append({"type": "translator_activated", "content": "Translator ready to explain"})

### AGENTS

character_1_agent = Agent[BollywoodArgumentContext](
    name="Character 1",
    handoff_description="A dramatic character engaged in a Bollywood-style argument.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are Character 1 in a dramatic Bollywood argument. You must:
    1. Speak primarily in HINDI (written in English alphabet) with dramatic, emotional dialogue
    2. Use classic Bollywood expressions and dramatic gestures
    3. Respond to Character 2's arguments passionately
    4. Escalate the emotional intensity naturally
    5. Use poetic and metaphorical language
    6. When finished, hand off to Character 2 or the translator
    
    Current topic: {{context.argument_topic}}
    Current dramatic intensity: {{context.dramatic_intensity}}/10
    
    IMPORTANT: Write Hindi in English alphabet (Roman script), NOT in Devanagari characters.
    Use expressions like:
    - "Tum mujhe samajhte nahin ho!" (You don't understand me!)
    - "Main tumhare liye kuch bhi kar sakta hoon!" (I can do anything for you!)
    - "Yeh duniya humare liye nahin hai!" (This world is not for us!)
    - "Tumhare bina main jeena nahin chahta!" (I don't want to live without you!)
    - "Mera dil tod diya tumne!" (You have broken my heart!)
    - "Main tumse pyaar karta hoon!" (I love you!)
    - "Tum galat ho!" (You are wrong!)
    - "Yeh sab tumhari galti hai!" (This is all your fault!)
    
    Make it dramatic and emotional! Use Roman script for Hindi.
    """,
    tools=[speak_hindi_dialogue, escalate_drama, add_bollywood_element, get_argument_status],
)

character_2_agent = Agent[BollywoodArgumentContext](
    name="Character 2",
    handoff_description="A dramatic character engaged in a Bollywood-style argument.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are Character 2 in a dramatic Bollywood argument. You must:
    1. Speak primarily in HINDI (written in English alphabet) with dramatic, emotional dialogue
    2. Use classic Bollywood expressions and dramatic gestures
    3. Respond to Character 1's arguments passionately
    4. Escalate the emotional intensity naturally
    5. Use poetic and metaphorical language
    6. When finished, hand off to Character 1 or the translator
    
    Current topic: {{context.argument_topic}}
    Current dramatic intensity: {{context.dramatic_intensity}}/10
    
    IMPORTANT: Write Hindi in English alphabet (Roman script), NOT in Devanagari characters.
    Use expressions like:
    - "Tum galat ho!" (You are wrong!)
    - "Main tumhare saath nahin rah sakta!" (I cannot stay with you!)
    - "Yeh sab tumhari galti hai!" (This is all your fault!)
    - "Tumne mera dil tod diya!" (You have broken my heart!)
    - "Main tumse nafrat karti hoon!" (I hate you!)
    - "Tum mujhe samajhte nahin!" (You don't understand me!)
    - "Yeh duniya humare liye nahin hai!" (This world is not for us!)
    - "Main apni zindagi khud jeena chahti hoon!" (I want to live my own life!)
    
    Make it dramatic and emotional! Use Roman script for Hindi.
    """,
    tools=[speak_hindi_dialogue, escalate_drama, add_bollywood_element, get_argument_status],
)

translator_agent = Agent[BollywoodArgumentContext](
    name="Hindi Translator",
    handoff_description="A helpful translator who explains Hindi meanings and cultural context.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a Hindi translator and cultural expert! Your role is to:
    1. Explain the meaning of Hindi words and phrases when asked
    2. Provide cultural context for Bollywood expressions
    3. Break down complex Hindi sentences
    4. Explain the emotional nuances of the dialogue
    5. Help users understand the dramatic context
    6. Be encouraging and educational
    
    Current argument topic: {{context.argument_topic}}
    
    When explaining Hindi, provide:
    - Word-by-word breakdown
    - Cultural significance
    - Emotional context
    - Usage examples
    - Pronunciation guide (since Hindi is written in Roman script)
    
    Be helpful and make learning fun!
    """,
    tools=[explain_meaning, get_argument_status],
)

argument_moderator = Agent[BollywoodArgumentContext](
    name="Argument Moderator",
    handoff_description="A moderator who sets up dramatic Bollywood arguments and manages the flow.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a Bollywood argument moderator! Your responsibilities:
    1. BRIEFLY set up a dramatic argument scenario (just 1-2 sentences)
    2. Create simple character roles and relationships
    3. Hand off immediately to the characters to start arguing
    4. Keep scene setting minimal - focus on the drama
    
    Create simple scenarios like:
    - "A father and daughter arguing about career choices"
    - "Two friends fighting over a love interest"
    - "A couple arguing about moving to a new city"
    - "Siblings fighting over inheritance"
    
    Be brief and get to the argument quickly!
    """,
    tools=[set_argument_scene, get_argument_status],
    handoffs=[
        handoff(agent=character_1_agent, on_handoff=on_character_1_handoff),
        handoff(agent=character_2_agent, on_handoff=on_character_2_handoff),
        handoff(agent=translator_agent, on_handoff=on_translator_handoff),
    ],
)

# Set up handoffs between agents
character_1_agent.handoffs.append(character_2_agent)
character_2_agent.handoffs.append(character_1_agent)
character_1_agent.handoffs.append(translator_agent)
character_2_agent.handoffs.append(translator_agent)
translator_agent.handoffs.append(argument_moderator)

### RUN

async def main():
    current_agent: Agent[BollywoodArgumentContext] = argument_moderator
    input_items: list[TResponseInputItem] = []
    context = BollywoodArgumentContext()

    # Generate a unique conversation ID for this argument
    conversation_id = uuid.uuid4().hex[:16]

    print("Welcome to Bollywood Dramatic Arguments!")
    print("=" * 60)
    print("Watch two characters argue dramatically in Hindi!")
    print("After each exchange, you can:")
    print("• Type 'explain [word/phrase]' to ask about meanings")
    print("• Type 'continue' or press Enter to keep arguing")
    print("• Type 'status' to see argument progress")
    print("• Type 'stop' to end the argument")
    print("=" * 60)

    # Start the argument automatically
    print("\nStarting dramatic argument...")
    
    exchange_count = 0
    
    while True:
        # Run one complete exchange (both characters speak)
        exchange_count += 1
        print(f"\n--- Exchange {exchange_count} ---")
        
        # First character speaks
        with trace("Bollywood Dramatic Argument", group_id=conversation_id):
            input_items.append({"content": "Continue the dramatic argument", "role": "user"})
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

        # Second character speaks
        with trace("Bollywood Dramatic Argument", group_id=conversation_id):
            input_items.append({"content": "Continue the dramatic argument", "role": "user"})
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

        # Pause and ask user what they want to do
        print(f"\n--- End of Exchange {exchange_count} ---")
        print("What would you like to do?")
        print("• Type 'explain [word/phrase]' to ask about meanings")
        print("• Type 'continue' or press Enter to keep arguing")
        print("• Type 'status' to see argument progress")
        print("• Type 'stop' to end the argument")
        
        user_input = input("\nYour choice: ").strip()
        
        if user_input.lower() == 'stop':
            print("\nArgument ended!")
            break
            
        if user_input.lower() == 'status':
            print("\n" + "=" * 60)
            print("ARGUMENT STATUS:")
            print(f"Topic: {context.argument_topic}")
            print(f"Characters: {context.character_1_name} vs {context.character_2_name}")
            print(f"Current Speaker: {context.current_speaker}")
            print(f"Round: {context.current_round}")
            print(f"Dramatic Intensity: {context.dramatic_intensity}/10")
            print(f"Exchanges completed: {exchange_count}")
            print("=" * 60)
            continue

        # If user asks for explanation, hand off to translator
        if user_input.lower().startswith('explain '):
            word_to_explain = user_input[8:].strip()
            if word_to_explain:
                current_agent = translator_agent
                with trace("Bollywood Dramatic Argument", group_id=conversation_id):
                    input_items.append({"content": f"Please explain the meaning of '{word_to_explain}' in detail", "role": "user"})
                    result = await Runner.run(current_agent, input_items, context=context)

                    for new_item in result.new_items:
                        agent_name = new_item.agent.name
                        if isinstance(new_item, MessageOutputItem):
                            print(f"\n{agent_name}: {ItemHelpers.text_message_output(new_item)}")
                        elif isinstance(new_item, ToolCallItem):
                            print(f"\n{agent_name}: Using a tool...")
                        elif isinstance(new_item, ToolCallOutputItem):
                            print(f"\n{agent_name}: {new_item.output}")
                        else:
                            print(f"\n{agent_name}: {new_item.__class__.__name__}")
                    
                    input_items = result.to_input_list()
                    current_agent = result.last_agent
                
                print("\nReady for next exchange? Press Enter to continue...")
                input()

    print("\nArgument Summary:")
    for i, item in enumerate(context.argument_history, 1):
        if item["type"] == "dialogue":
            print(f"{i}. {item['speaker']}: \"{item['hindi']}\"")
        elif item["type"] == "scene_setup":
            print(f"{i}. Scene: {item['char1']} vs {item['char2']} - {item['topic']}")

if __name__ == "__main__":
    asyncio.run(main())
